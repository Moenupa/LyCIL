import copy
from tqdm import tqdm
from abc import abstractmethod
from typing import Literal, Optional, Any, Union

from datasets import Dataset

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

from ..backbone import BaseBackbone, ConvNetArgs, ResNetBackbone
from ..classifier import expand_head, make_head
from ..constants import _X_COLUMN_NAME, _Y_COLUMN_NAME
from ..data.buffer import compute_nme
from ..data.hfmodule import filter_by_classid
from ..metrics.accuracy import accuracy, accuracy_topk
from ..scheduler import LinearWarmupCosineAnnealingLR

import torch.nn as nn

from ..data.buffer import BaseExemplarBuffer
from ..data.hfmodule import HFDataModule
from ..data.transform import apply_dataset_transform

from ..optimizer import LARS






class BaseLearner(L.LightningModule):
    r"""Base class providing backbone, classifier head, optimizer, and memory helpers.

        Subclasses must implement:

        - :meth:`training_step` — define the per-batch loss.
        - :meth:`on_train_end` — triggered after each task's training loop; override
          to call :meth:`update_memory` for replay-based methods. The default
          implementation does nothing (no memory update).

        Args:
            backbone_cls (type[BaseBackbone], optional): Backbone class to
                instantiate. (default: :class:`~lycil.backbone.ResNetBackbone`)
            backbone_args (ConvNetArgs | None, optional): Arguments forwarded to
                ``backbone_cls``. Uses :class:`~lycil.backbone.ConvNetArgs` defaults
                if ``None``. (default: ``None``)
            head (Literal["linear", "cosine"], optional): Classifier head type.
                (default: ``"linear"``)
            data_column_translate (dict[str, str] | None, optional): Column name
                remapping applied when reading batches. (default: ``None``)
            per_task_optim_args (dict[int, dict] | None, optional): Per-task
                optimizer keyword arguments keyed by task ID. Use ``-1`` as a
                fallback key. (default: ``None``)
            per_task_sched_args (dict[int, dict] | None, optional): Per-task
                scheduler keyword arguments keyed by task ID. Use ``-1`` as a
                fallback key. (default: ``None``)
        """

    def __init__(
            self,
            *,
            backbone_cls: "type[BaseBackbone]" = ResNetBackbone,
            backbone_args: ConvNetArgs | None = None,
            head: Literal["linear", "cosine"] = "linear",
            data_column_translate: dict[str, str] | None = None,
            per_task_optim_args: dict[Union[int, str], dict] | None = None,
            per_task_sched_args: dict[Union[int, str], dict] | None = None,
            buffer_args: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone_cls(backbone_args or ConvNetArgs())
        self.head_type = head
        # lazy init by head_type at `expand_head()`
        self.classifier: nn.Module | None = None

        self.buffer: BaseExemplarBuffer | None = None
        self._old_self: BaseLearner | None = None

        # lazy init by `set_task_id()` to sync with data module
        self.task_id: int = None
        self.num_old_classes: int = None
        self.num_seen_classes: int = None

        self.data_column_translate: dict[str, str] = data_column_translate or {}
        # kwargs for optimizer/scheduler per task_id
        # e.g. {0: {"type":"sgd", "lr":0.1}, 1: {"type":"sgd", "lr":0.01}}
        # first task SGD(lr=0.1), second task SGD(lr=0.01)
        self.per_task_optim_args: dict[Union[int, str], dict] = per_task_optim_args or {}
        self.per_task_sched_args: dict[Union[int, str], dict] = per_task_sched_args or {}

        self.buffer_args: dict[str, Any] = {
            "selection": "herding",
            "seed": 42,
            "loader_kwargs": {
                "batch_size": 128,
                "shuffle": False,
                "num_workers": 8,
            },
            "nme_eval": {
                "enable": True,
                "topk": 1,
                "dynamic_old": True,
                "dynamic_new": True,
            },
        }

        if buffer_args is not None:
            self.buffer_args.update(copy.deepcopy(buffer_args))
            if "nme_eval" in buffer_args:
                self.buffer_args["nme_eval"].update(copy.deepcopy(buffer_args["nme_eval"]))

        self._cached_val_nme = None
        self._cached_test_nme = None

    @property
    def feature_dim(self) -> int:
        """Expose backbone feature dimension used by classifier heads."""
        return self.backbone.feature_dim

    def set_task_id(self, task_id: int):
        self.task_id = task_id

    def sync_with_datamodule(self, dm: "HFDataModule"):
        """Synchronize task state and expand the classifier head to match the data module.

        Reads the current task ID and class counts from ``dm`` and, when the data
        module has advanced to a new task, expands the classifier head by the
        appropriate number of new classes.

        A negative ``task_id`` acts as a bypass: head expansion is skipped,
        which is useful for multi-stage per-task training (e.g., first stage
        without buffer, second stage with buffer).

        Args:
            dm (HFDataModule): Data module to sync from.

        Raises:
            RuntimeError: If the data module reports more seen classes than
                expected, indicating a call-order bug.
        """
        dm_task_id = dm.get_current_task()
        if self.task_id is not None and dm_task_id == self.task_id:
            # in sync, no update
            return
        if self.task_id is not None and self.task_id < 0:
            # a special bypass rule for multi-stage training per task,
            # e.g. first stage with no buffer, second stage with buffer.
            # this will disable head expansion, to do this, you should:
            # manually set `learner.set_task_id(-2)`
            # and reset `learner.set_task_id(cur_task_id)`
            return

        self.task_id = dm_task_id

        incoming_expansion = dm.num_seen_classes - (self.num_seen_classes or 0)
        if incoming_expansion <= 0:
            raise RuntimeError(
                f"Expect an incoming expansion, got {incoming_expansion} new classes. "
                + f"Data has {dm.num_seen_classes} seen classes, "
                + f"but Model has {self.num_seen_classes} seen classes. "
                + "Ensure that `sync_with_datamodule()` is called after datamodule updates."
            )

        self.expand_head(incoming_expansion)

        self.num_old_classes = self.num_seen_classes or 0
        self.num_seen_classes = dm.num_seen_classes

    @staticmethod
    def unpack_batch(
            batch: dict[str, torch.Tensor],
            device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the input tensor and label tensor from a batch dict.

        Args:
            batch (dict[str, torch.Tensor]): Batch dict from a dataloader,
                keyed by internal column names.
            device (torch.device | None, optional): If provided, moves both
                tensors to this device before returning. (default: ``None``)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(x, y)`` — input images and
                integer class labels.
        """
        x = batch[_X_COLUMN_NAME]
        y = batch[_Y_COLUMN_NAME]
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        return x, y

    @torch.no_grad()
    def expand_head(self, out_delta: int, in_delta: int = 0) -> None:
        """Initialize or expand the classifier head to accommodate new classes.

        Args:
            out_delta (int):
                Number of new classes (i.e. out features).
            in_delta (int, optional):
                Number of new input features, only used for DERNet-like structure. (default: 0)

        """
        if self.classifier is None:
            self.classifier = make_head(
                self.feature_dim, out_delta, head_type=self.head_type
            )
            return

        # usually, expand number of new classes (i.e. out_delta)
        # infeatures dont have to expand unless dernet-like structure is defined
        self.classifier = expand_head(
            self.classifier, out_delta=out_delta, in_delta=in_delta
        )
        return

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_layerwise(x)["features"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_layerwise(x)["logits"]

    @torch.no_grad()
    def forward_no_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass without gradient tracking.

        Convenience wrapper used when computing distillation targets or
        exemplar statistics from a frozen model snapshot.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            torch.Tensor: Logit tensor of shape ``(batch_size, num_seen_classes)``.
        """
        return self.forward(x)

    def forward_layerwise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a full forward pass and return backbone intermediates plus classifier outputs.

        Merges the dict returned by
        :meth:`~lycil.backbone.BaseBackbone.forward_layerwise` with the dict
        returned by the classifier head, so the result contains both feature
        maps and logits under a single namespace.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            dict[str, torch.Tensor]: Combined dict with backbone intermediate
                keys (e.g., ``"l1"``–``"l4"``, ``"features"``) and classifier
                outputs (e.g., ``"logits"``).

        Raises:
            RuntimeError: If the classifier head has not been initialized yet.
                Call :meth:`expand_head` before the first forward pass.
        """
        if self.classifier is None:
            raise RuntimeError(
                "Classifier head is not initialized. Call expand_head before training."
            )

        fmap = self.backbone.forward_layerwise(x)
        logits: dict[str, torch.Tensor] = self.classifier(fmap["features"])
        fmap.update(logits)
        # with keys 'l1', 'l2', 'l3', 'l4', 'features', 'logits'
        return fmap

    @staticmethod
    def _get_optimizer(*args, **kwargs):
        opt_type = kwargs.pop("type", "sgd")
        match opt_type:
            case "sgd":
                return torch.optim.SGD(*args, **kwargs)
            case "adam":
                return torch.optim.Adam(*args, **kwargs)
            case "adamw":
                return torch.optim.AdamW(*args, **kwargs)
            case "lars":
                return LARS(*args, **kwargs)
            case _:
                raise NotImplementedError(f"Unsupported optimizer: `{opt_type}`")

    @staticmethod
    def _get_scheduler(*args, **kwargs):
        sched_type = kwargs.pop("type", "linear_warmup_cosine_annealing")
        match sched_type:
            case "linear_warmup_cosine_annealing":
                return LinearWarmupCosineAnnealingLR(*args, **kwargs)
            case "cosine_annealing":
                return lr_scheduler.CosineAnnealingLR(*args, **kwargs)
            case "step_lr":
                return lr_scheduler.StepLR(*args, **kwargs)
            case "multi_step_lr":
                return lr_scheduler.MultiStepLR(*args, **kwargs)
            case _:
                raise NotImplementedError(f"Unsupported scheduler: `{sched_type}`")

    # def configure_optimizers(self):
    #     params = [p for p in self.parameters() if p.requires_grad]
    #
    #     # Select stage-specific key for optimizer/scheduler configs.
    #
    #     # Waterfall lookup: stage_key -> default -> {}
    #     optim_kwargs = (
    #             self.per_task_optim_args.get(self.task_id)
    #             or self.per_task_optim_args.get("default")
    #             or {}
    #     )
    #     sched_kwargs = (
    #             self.per_task_sched_args.get(self.task_id)
    #             or self.per_task_sched_args.get("default")
    #             or {}
    #     )
    #
    #     optim = self._get_optimizer(params, **optim_kwargs)
    #     # If sched_kwargs is None (or explicitly disabled), return optimizer only
    #     if not sched_kwargs or sched_kwargs.get("type") in (None, "none", "None"):
    #         return optim
    #
    #     sched = self._get_scheduler(optim, **sched_kwargs)
    #     return {
    #         "optimizer": optim,
    #         "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
    #     }

    def _get_optim_stage_key(self):
        return self.task_id

    def _build_param_groups(self, weight_decay: float):
        decay, no_decay, no_decay_names = [], [], []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # n = name.lower()
            # if name.endswith(".bias") or ".bn" in n or "bn." in n or "norm" in n:
            #     no_decay.append(p)
            #     no_decay_names.append(name)
            # else:
            #     decay.append(p)
            decay.append(p)
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        grouped = sum(p.numel() for p in decay) + sum(p.numel() for p in no_decay)
        assert total == grouped, f"Param mismatch: total={total}, grouped={grouped}"

        print("[No Weight Decay Params]")
        print("\n".join(no_decay_names))

        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        stage_key = self._get_optim_stage_key()
        optim_kwargs = dict(
            self.per_task_optim_args.get(stage_key)
            or self.per_task_optim_args.get("default")
            or {}
        )
        sched_kwargs = dict(
            self.per_task_sched_args.get(stage_key)
            or self.per_task_sched_args.get("default")
            or {}
        )

        weight_decay = float(optim_kwargs.pop("weight_decay", 0.0) or 0.0)
        params = self._build_param_groups(weight_decay)

        optim = self._get_optimizer(params, **optim_kwargs)

        if not sched_kwargs or sched_kwargs.get("type") in (None, "none", "None"):
            return optim

        sched = self._get_scheduler(optim, **sched_kwargs)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }



    @torch.no_grad()
    def snapshot_old(self):
        """Store a frozen deep copy of the current model as the old-task reference.

        The snapshot is set to eval mode and all its parameters have
        ``requires_grad=False``. Called automatically by :meth:`on_fit_end`
        after each task.
        """
        # prevent recursive copies
        self._old_self = None

        # snapshot and freeze
        self._old_self = copy.deepcopy(self).eval()
        for p in self._old_self.parameters():
            p.requires_grad_(False)

    @property
    def old_self(self) -> "BaseLearner":
        """Frozen snapshot of the model after the previous task.

        Updated by :meth:`snapshot_old` (called automatically in
        :meth:`on_fit_end`). Useful for knowledge-distillation losses.

        Raises:
            RuntimeError: If :meth:`snapshot_old` has not been called yet.
        """
        if self._old_self is None:
            raise RuntimeError(
                "No old model snapshot stored. Call `snapshot_old()` first."
            )
        return self._old_self

    def setup(self, stage) -> None:
        super().setup(stage)
        if stage == "fit":
            dm: HFDataModule = self.trainer.datamodule  # ty: ignore[unresolved-attribute]
            self.sync_with_datamodule(dm)

    def on_fit_end(self):
        self.snapshot_old()

    @abstractmethod
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        ...


    @torch.no_grad()
    def update_memory(self, dm: HFDataModule, **kwargs) -> None:
        """Update datamodule's exemplar memory (i.e., iCaRL).

        To be opted-in after training of each task, e.g. in ``on_train_end()``.

        Args:
            dm (HFDataModule): The data module containing the buffer.
            kwargs: Additional arguments for exemplar construction.

        Raises:
            RuntimeError: If the buffer is not initialized.
        """
        if dm.buffer is None:
            raise RuntimeError("Buffer is not initialized.")

        per_class_quota = dm.buffer.size_per_class(self.num_seen_classes)

        self.eval()
        if dm.buffer.is_adaptive:
            # Adaptive memory: reuse the full global budget on the currently
            # seen classes, then shrink old classes as new ones arrive.
            dm.buffer.reduce_exemplars(per_class_quota)
        self._construct_exemplar_unified(dm, per_class_quota=per_class_quota, **kwargs, )
        self.train()
        return

    @torch.no_grad()
    def _construct_exemplar(self, dm: "HFDataModule", **kwargs) -> None:
        """Construct class exemplars for adaptive-memory strategies.

        Args:
            dm (HFDataModule): Data module to update, containing the buffer.
            kwargs: Additional arguments for exemplar construction.

        """
        raise NotImplementedError

        assert dm.buffer is not None
        # construct exemplar set for current classes
        for class_idx in range(self.num_old_classes, self.num_seen_classes):
            pass

    @torch.no_grad()
    def _construct_exemplar_unified(self, dm: HFDataModule, **kwargs) -> None:
        """
        统一的 exemplar 构建实现。

        同时支持两种 buffer 策略：
        - 总内存自适应模式（`mem_size`）
        - 固定每类样本数模式（`mem_size_per_class`）

        调用约定：
        - 自适应总内存模式下，应当在调用本函数前先把旧类 exemplar 裁剪到新的配额；
        - 固定每类样本数模式下，可以直接调用本函数，对旧类重算 mean，并为新类构建 exemplar。
        """

        assert dm.buffer is not None

        loader_kwargs = kwargs.get(
            "loader_kwargs",
            self.buffer_args.get(
                "loader_kwargs",
                {"batch_size": 128, "shuffle": False, "num_workers": 8},
            ),
        )

        exemplar_selection = kwargs.get(
            "exemplar_selection",
            self.buffer_args.get("selection", "herding"),
        )

        exemplar_seed = int(
            kwargs.get(
                "exemplar_seed",
                self.buffer_args.get("seed", 42),
            )
        )


        # 当前每个类别允许保留的 exemplar 数量
        # 对于 mem_size 模式，这里会根据当前已见类别数动态计算
        per_class_quota = int(
            kwargs.get(
                "per_class_quota",
                dm.buffer.size_per_class(self.num_seen_classes),
            )
        )

        # 统一使用 test transform 提特征，避免随机增强影响 exemplar 选择和类均值计算
        feature_tfm = dm.get_effective_transform(mode="test")

        if exemplar_selection not in {"random", "herding"}:
            raise ValueError(
                f"Unsupported exemplar_selection={exemplar_selection}, "
                "expected one of {'random', 'herding'}."
            )

        per_class_means = {}

        # 1) 用最新网络重新计算旧类别的类中心
        for class_idx in range(self.num_old_classes):
            if f"{class_idx}" not in dm.buffer:
                continue
            if len(dm.buffer[f"{class_idx}"]) == 0:
                continue

            loader = dm.buffer.get_dataloader(
                keys=[f"{class_idx}"],
                transform=feature_tfm,
                loader_kwargs=loader_kwargs,
            )
            mean, _ = compute_nme(loader, self.feature_extractor, self.device)
            mean = F.normalize(mean, dim=0)
            per_class_means[class_idx] = mean.cpu()

        # 2) 先一次性筛出当前 task 的训练子集：
        #    - raw 版本用于真正写入 buffer，必须保留原始 HF features/schema；
        #    - feat 版本仅用于提特征做 herding。
        task_train_dataset_raw = dm.get_filtered_dataset(
            split=dm._split_train,
            filter_fn=lambda e: self.num_old_classes <= e[_Y_COLUMN_NAME] < self.num_seen_classes,
            transform=None,
            use_buffer=False,
        )
        task_train_dataset_feat = dm.get_filtered_dataset(
            split=dm._split_train,
            filter_fn=lambda e: self.num_old_classes <= e[_Y_COLUMN_NAME] < self.num_seen_classes,
            transform=feature_tfm,
            use_buffer=False,
        )

        # 3) 在当前 task 子集内，先建立 “类别 -> 样本索引列表” 的映射
        #    这样后面每个类别直接 select 对应索引即可，不再重复 filter
        class_to_indices = {class_idx: [] for class_idx in range(self.num_old_classes, self.num_seen_classes)}
        task_labels = task_train_dataset_raw[_Y_COLUMN_NAME]
        for sample_idx, y in enumerate(task_labels):
            class_to_indices[int(y)].append(sample_idx)

        # 4) 为当前新类别构建 exemplar 集合
        # for class_idx in range(self.num_old_classes, self.num_seen_classes):
        for class_idx in tqdm(
                range(self.num_old_classes, self.num_seen_classes),
                desc=f"Building exemplars task {dm.get_current_task()}",
        ):
            class_indices = class_to_indices.get(class_idx, [])
            if len(class_indices) == 0:
                continue

            # 从当前 task 子集中切出该类别的数据
            class_dataset = task_train_dataset_raw.select(class_indices)
            class_dataset_feat = task_train_dataset_feat.select(class_indices)

            n_samples = len(class_dataset)
            if n_samples == 0:
                continue

            # 当前类实际保留的样本数，不能超过该类真实样本数
            m = min(per_class_quota, n_samples)
            selected_idx = self._select_exemplar_indices(
                class_dataset_feat=class_dataset_feat,
                m=m,
                loader_kwargs=loader_kwargs,
                exemplar_selection=exemplar_selection,
                exemplar_seed=exemplar_seed,
                class_idx=class_idx,
            )

            # 5) 将选中的 exemplar 写入 buffer
            #    必须写入 raw dataset，避免把 transform/view 混进 buffer。
            selected_dataset = class_dataset.select(selected_idx)
            selected_dataset.reset_format()
            dm.buffer[f"{class_idx}"] = selected_dataset

            # 6) 用选中的 exemplar 重新计算该类的类中心
            loader = dm.buffer.get_dataloader(
                keys=[f"{class_idx}"],
                transform=feature_tfm,
                loader_kwargs=loader_kwargs,
            )
            mean, _ = compute_nme(loader, self.feature_extractor, self.device)
            mean = F.normalize(mean, dim=0)
            per_class_means[class_idx] = mean.cpu()

        # 保存所有类别的类中心，供 NME 推理使用
        dm.buffer.per_class_means = per_class_means

        return


    # @torch.no_grad()
    # def _select_exemplar_indices(
    #         self,
    #         class_dataset_feat: Dataset,
    #         m: int,
    #         loader_kwargs: dict,
    #         exemplar_selection: str,
    #         exemplar_seed: int,
    #         class_idx: int,
    # ) -> list[int]:
    #     n_samples = len(class_dataset_feat)
    #     if n_samples == 0 or m <= 0:
    #         return []
    #
    #     m = min(m, n_samples)
    #
    #     if exemplar_selection == "random":
    #         g = torch.Generator()
    #         g.manual_seed(exemplar_seed + int(class_idx))
    #         return torch.randperm(n_samples, generator=g)[:m].tolist()
    #
    #     if exemplar_selection != "herding":
    #         raise ValueError(f"Unsupported exemplar_selection={exemplar_selection}")
    #
    #     loader = torch.utils.data.DataLoader(class_dataset_feat, **loader_kwargs)
    #     class_mean, per_sample_features = compute_nme(
    #         loader, self.feature_extractor, self.device
    #     )
    #
    #     class_mean = F.normalize(class_mean, dim=0)
    #     feats = per_sample_features  # [N, D]
    #
    #     # 如果 compute_nme 返回的 per_sample_features 还没归一化，可以打开这一行
    #     # feats = F.normalize(feats, dim=1)
    #
    #     # 预先算好每个样本的平方范数，后面循环里复用
    #     feat_sq_norm = (feats * feats).sum(dim=1)  # [N]
    #
    #     selected_idx = []
    #     selected_mask = torch.zeros(n_samples, dtype=torch.bool, device=feats.device)
    #     running_sum = torch.zeros_like(class_mean)
    #
    #     for k in range(1, m + 1):
    #         # 原目标：
    #         #   argmin_i || class_mean - (running_sum + f_i) / k ||_2
    #         #
    #         # 等价于：
    #         #   argmin_i || k * class_mean - running_sum - f_i ||_2
    #         #
    #         # 再展开平方范数，可转成最大化：
    #         #   2 * <target, f_i> - ||f_i||^2
    #         target = k * class_mean - running_sum  # [D]
    #
    #         scores = 2.0 * (feats @ target) - feat_sq_norm  # [N]
    #         scores.masked_fill_(selected_mask, float("-inf"))
    #
    #         best_abs = int(scores.argmax().item())
    #         selected_idx.append(best_abs)
    #         selected_mask[best_abs] = True
    #         running_sum += feats[best_abs]
    #
    #     return selected_idx

    @torch.no_grad()
    def _select_exemplar_indices(
            self,
            class_dataset_feat: Dataset,
            m: int,
            loader_kwargs: dict,
            exemplar_selection: str,
            exemplar_seed: int,
            class_idx: int,
    ) -> list[int]:
        n_samples = len(class_dataset_feat)
        if n_samples == 0 or m <= 0:
            return []

        if exemplar_selection == "random":
            g = torch.Generator()
            g.manual_seed(exemplar_seed + int(class_idx))
            return torch.randperm(n_samples, generator=g)[:m].tolist()

        if exemplar_selection != "herding":
            raise ValueError(f"Unsupported exemplar_selection={exemplar_selection}")

        loader = torch.utils.data.DataLoader(class_dataset_feat, **loader_kwargs)
        class_mean, per_sample_features = compute_nme(
            loader, self.feature_extractor, self.device
        )

        class_mean = F.normalize(class_mean, dim=0)
        feats = per_sample_features

        selected_idx = []
        selected_mask = torch.zeros(n_samples, dtype=torch.bool, device=feats.device)
        running_sum = torch.zeros_like(class_mean)

        for k in range(1, min(m, n_samples) + 1):
            candidate_idx = (~selected_mask).nonzero(as_tuple=False).squeeze(1)
            candidate_feats = feats[candidate_idx]

            mu_p = (running_sum.unsqueeze(0) + candidate_feats) / k
            dist = torch.norm(class_mean.unsqueeze(0) - mu_p, p=2, dim=1)
            best_rel = torch.argmin(dist).item()
            best_abs = candidate_idx[best_rel].item()

            selected_idx.append(best_abs)
            selected_mask[best_abs] = True
            running_sum += feats[best_abs]

        return selected_idx

    @torch.no_grad()
    def _build_eval_nme_state(
            self,
            dm: HFDataModule,
            include_new_tmp: bool,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        # import pdb;pdb.set_trace()
        if dm.buffer is None:
            return None

        loader_kwargs = self.buffer_args.get(
            "loader_kwargs",
            {"batch_size": 128, "shuffle": False, "num_workers": 8},
        )

        dynamic_old = bool(self.buffer_args["nme_eval"].get("dynamic_old", True))

        exemplar_selection = str(self.buffer_args.get("selection", "herding"))
        exemplar_seed = int(self.buffer_args.get("seed", 42))

        feature_tfm = dm.get_effective_transform(mode="test")
        per_class_quota = dm.buffer.size_per_class(self.num_seen_classes)

        per_class_means: dict[int, torch.Tensor] = {}

        # old classes: 用当前模型动态重算 buffer exemplar 的中心
        for class_idx in range(self.num_old_classes):
            key = f"{class_idx}"
            if key not in dm.buffer or len(dm.buffer[key]) == 0:
                continue

            if dynamic_old:
                old_dataset = dm.buffer[key]
                old_dataset.reset_format()
                apply_dataset_transform(old_dataset, transform=feature_tfm)

                loader = torch.utils.data.DataLoader(old_dataset, **loader_kwargs)
                mean, _ = compute_nme(loader, self.feature_extractor, self.device)
                per_class_means[class_idx] = F.normalize(mean, dim=0).cpu()
            else:
                if class_idx in dm.buffer.per_class_means:
                    per_class_means[class_idx] = dm.buffer.per_class_means[class_idx].cpu()

        # new classes: 先做 tmp selecting，再算 tmp mean
        if include_new_tmp and self.num_seen_classes > self.num_old_classes:
            task_train_dataset_raw = dm.get_filtered_dataset(
                split=dm._split_train,
                filter_fn=lambda e: self.num_old_classes <= e[_Y_COLUMN_NAME] < self.num_seen_classes,
                transform=None,
                use_buffer=False,
            )
            task_train_dataset_feat = dm.get_filtered_dataset(
                split=dm._split_train,
                filter_fn=lambda e: self.num_old_classes <= e[_Y_COLUMN_NAME] < self.num_seen_classes,
                transform=feature_tfm,
                use_buffer=False,
            )

            class_to_indices = {
                class_idx: [] for class_idx in range(self.num_old_classes, self.num_seen_classes)
            }
            task_labels = task_train_dataset_raw[_Y_COLUMN_NAME]


            for sample_idx, y in enumerate(task_labels):
                class_to_indices[int(y)].append(sample_idx)

            for class_idx in tqdm(
                    range(self.num_old_classes, self.num_seen_classes),
                    desc=f"Building TMP exemplars task {dm.get_current_task()} for NME evaluation",
            ):
                class_indices = class_to_indices.get(class_idx, [])
                if len(class_indices) == 0:
                    continue

                class_dataset_raw = task_train_dataset_raw.select(class_indices)
                class_dataset_feat = task_train_dataset_feat.select(class_indices)

                m = min(per_class_quota, len(class_dataset_raw))
                selected_idx = self._select_exemplar_indices(
                    class_dataset_feat=class_dataset_feat,
                    m=m,
                    loader_kwargs=loader_kwargs,
                    exemplar_selection=exemplar_selection,
                    exemplar_seed=exemplar_seed,
                    class_idx=class_idx,
                )

                tmp_dataset = class_dataset_raw.select(selected_idx)
                tmp_dataset.reset_format()
                apply_dataset_transform(tmp_dataset, transform=feature_tfm)

                loader = torch.utils.data.DataLoader(tmp_dataset, **loader_kwargs)
                mean, _ = compute_nme(loader, self.feature_extractor, self.device)
                per_class_means[class_idx] = F.normalize(mean, dim=0).cpu()


        if len(per_class_means) == 0:
            return None

        class_ids = sorted(per_class_means.keys())
        class_ids_t = torch.tensor(class_ids, dtype=torch.long)
        class_means_t = torch.stack([per_class_means[c] for c in class_ids], dim=0)
        class_means_t = F.normalize(class_means_t, dim=1)

        return class_ids_t, class_means_t

    def on_validation_epoch_start(self) -> None:
        if not self._should_run_val_nme_eval():
            self._cached_val_nme = None
            return

        dm: HFDataModule = self.trainer.datamodule
        include_new_tmp = bool(self.buffer_args["nme_eval"].get("dynamic_new", True))
        self._cached_val_nme = self._build_eval_nme_state(
            dm,
            include_new_tmp=include_new_tmp,
        )

    def on_validation_epoch_end(self) -> None:
        self._cached_val_nme = None

    def on_test_epoch_start(self) -> None:
        # import pdb;pdb.set_trace()
        if not self.buffer_args["nme_eval"].get("enable", True):
            self._cached_test_nme = None
            return

        dm: HFDataModule = self.trainer.datamodule
        self._cached_test_nme = self._build_eval_nme_state(
            dm,
            include_new_tmp=True,
        )

    def on_test_epoch_end(self) -> None:
        self._cached_test_nme = None

    @torch.no_grad()
    def _predict_nme_rank(
            self,
            x: torch.Tensor,
            class_ids: torch.Tensor,
            class_means: torch.Tensor,
            topk: int = 1,
    ) -> torch.Tensor:
        feats = F.normalize(self.feature_extractor(x), dim=1)
        means = F.normalize(class_means.to(feats.device), dim=1)
        class_ids = class_ids.to(feats.device)

        dists = torch.cdist(feats, means, p=2).pow(2)
        rank = torch.argsort(dists, dim=1)[:, :topk]
        return class_ids[rank]

    @staticmethod
    def _rank_top1_acc(rank_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return (rank_pred[:, 0] == y_true).float().mean()

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        x, y = self.unpack_batch(batch)

        logits: torch.Tensor = self(x)
        acc1 = accuracy(logits, y)

        dm = self.trainer.datamodule
        name = getattr(dm, "_val_loader_names", None)
        suffix = name[dataloader_idx] if name is not None else f"dl{dataloader_idx}"

        self.log(
            name=f"val_{suffix}",
            value=acc1,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if self._cached_val_nme is not None:
            topk = int(self.buffer_args["nme_eval"].get("topk", 1))
            class_ids, class_means = self._cached_val_nme
            rank_pred = self._predict_nme_rank(x, class_ids, class_means, topk=topk)
            nme_acc1 = self._rank_top1_acc(rank_pred, y)

            self.log(
                name=f"val_nme_{suffix}",
                value=nme_acc1,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        x, y = self.unpack_batch(batch)

        logits: torch.Tensor = self(x)
        acc1 = accuracy(logits, y)

        dm = self.trainer.datamodule
        name = getattr(dm, "_test_loader_names", None)
        suffix = name[dataloader_idx] if name is not None else f"dl{dataloader_idx}"

        self.log(
            name=f"test_{suffix}",
            value=acc1,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if self._cached_test_nme is not None:
            topk = int(self.buffer_args["nme_eval"].get("topk", 1))
            class_ids, class_means = self._cached_test_nme
            rank_pred = self._predict_nme_rank(x, class_ids, class_means, topk=topk)
            nme_acc1 = self._rank_top1_acc(rank_pred, y)

            self.log(
                name=f"test_nme_{suffix}",
                value=nme_acc1,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def _should_run_val_nme_eval(self) -> bool:
        nme_cfg = self.buffer_args.get("nme_eval", {})
        trainer = self.trainer
        dm = trainer.datamodule

        # # 1) 未启用 buffer 时，直接跳过 NME 评估
        if not (dm.buffer is not None and len(dm.buffer) > 0):
            return False

        # 2) NME 评估总开关关闭时，直接不评估
        if not nme_cfg.get("enable", True):
            return False

        # 3) sanity checking 阶段不评估
        if getattr(trainer, "sanity_checking", False):
            return False

        every_n_epochs = max(1, int(nme_cfg.get("every_n_epochs", 20)))

        # Lightning 的 current_epoch 是 0-based，这里转成 1-based 更直观
        epoch_idx = trainer.current_epoch + 1
        max_epochs = getattr(trainer, "max_epochs", None)

        # 4) 最后一个 epoch 必定评估
        is_last_epoch = (
                isinstance(max_epochs, int)
                and 0 < max_epochs <= epoch_idx
        )

        # 5) 普通情况下每隔 every_n_epochs 评估一次
        is_interval_epoch = (epoch_idx % every_n_epochs == 0)

        return is_interval_epoch or is_last_epoch