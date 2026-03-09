import copy
from tqdm import tqdm
from abc import abstractmethod
from typing import Literal, Optional

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
        per_task_optim_args: dict[int, dict] | None = None,
        per_task_sched_args: dict[int, dict] | None = None,
    ):
        super().__init__()

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
        self.per_task_optim_args: dict[int, dict] = per_task_optim_args or {}
        self.per_task_sched_args: dict[int, dict] = per_task_sched_args or {}

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
    def expand_head(self, num_new: int) -> None:
        """Initialize or expand the classifier head to accommodate new classes.

        On the first call (when ``self.classifier`` is ``None``), creates a
        fresh head with ``num_new`` outputs. On subsequent calls, delegates to
        :func:`~lycil.classifier.expand_head` to grow the existing head.

        Args:
            num_new (int): Number of new output classes to add.
        """
        if self.classifier is None:
            self.classifier = make_head(
                self.feature_dim, num_new, head_type=self.head_type
            )
            return

        self.classifier = expand_head(self.classifier, num_new)
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
            case "adamw":
                return torch.optim.AdamW(*args, **kwargs)
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

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]

        # a waterfall lookup for optimizer/scheduler kwargs:
        # per-task specific > default (-1) > empty dict
        optim_kwargs = (
            self.per_task_optim_args.get(self.task_id)
            or self.per_task_optim_args.get(-1)
            or {}
        )
        sched_kwargs = (
            self.per_task_sched_args.get(self.task_id)
            or self.per_task_sched_args.get(-1)
            or {}
        )
        optim = self._get_optimizer(params, **optim_kwargs)
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
    def training_step(self, batch, batch_idx: int) -> torch.Tensor: ...

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        x, y = self.unpack_batch(batch)
        logits: torch.Tensor = self(x)
        acc1 = accuracy(logits, y)

        dm = self.trainer.datamodule  # HFDataModule
        name = getattr(dm, "_val_loader_names", None)
        suffix = name[dataloader_idx] if name is not None else f"dl{dataloader_idx}"

        self.log(
            name=f"val_{suffix}",
            value=acc1,
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
        # exemplar 构建时使用的 dataloader 配置
        # 这里固定不用 shuffle，避免 exemplar 选择过程不稳定
        loader_kwargs = dict(
            batch_size=128,
            shuffle=False,
            num_workers=8,
        )

        assert dm.buffer is not None

        # exemplar 选择策略，默认 herding
        exemplar_selection = kwargs.get(
            "exemplar_selection",
            getattr(self, "exemplar_selection", "herding"),
        )
        # 随机选择时使用的随机种子
        exemplar_seed = int(
            kwargs.get(
                "exemplar_seed",
                getattr(self, "exemplar_seed", 42),
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
            mean = F.normalize(mean.unsqueeze(0), dim=1).squeeze(0)
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

            # 4.1 随机选择 exemplar
            if exemplar_selection == "random":
                g = torch.Generator()
                g.manual_seed(exemplar_seed + int(class_idx))
                selected_idx = torch.randperm(n_samples, generator=g)[:m].tolist()

            # 4.2 herding 选择 exemplar
            else:
                # 直接在当前类别子集上做一次前向，提取特征
                train_loader = torch.utils.data.DataLoader(class_dataset_feat, **loader_kwargs)
                class_mean, per_sample_features = compute_nme(
                    train_loader, self.feature_extractor, self.device
                )

                class_mean = F.normalize(class_mean.unsqueeze(0), dim=1).squeeze(0).cpu()
                feats = per_sample_features.cpu()

                selected_idx = []
                selected_mask = torch.zeros(n_samples, dtype=torch.bool)
                running_sum = torch.zeros_like(class_mean)

                # 按 herding 规则逐个选择，使 exemplar 均值尽量逼近类中心
                for k in range(1, m + 1):
                    candidate_idx = (~selected_mask).nonzero(as_tuple=False).squeeze(1)
                    candidate_feats = feats[candidate_idx]

                    mu_p = (running_sum.unsqueeze(0) + candidate_feats) / k
                    dist = torch.norm(class_mean.unsqueeze(0) - mu_p, p=2, dim=1)
                    best_rel = torch.argmin(dist).item()
                    best_abs = candidate_idx[best_rel].item()

                    selected_idx.append(best_abs)
                    selected_mask[best_abs] = True
                    running_sum += feats[best_abs]

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
            mean = F.normalize(mean.unsqueeze(0), dim=1).squeeze(0)
            per_class_means[class_idx] = mean.cpu()

        # 保存所有类别的类中心，供 NME 推理使用
        dm.buffer.per_class_means = per_class_means

        return