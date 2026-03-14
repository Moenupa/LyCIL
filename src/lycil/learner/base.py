import copy
from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Literal

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

from ..backbone import BaseBackbone, ConvNetArgs, ResNetBackbone
from ..classifier import expand_head, make_head
from ..constants import _X_COLUMN_NAME, _Y_COLUMN_NAME
from ..data.buffer import (
    BufferReplayArgs,
    compute_nme,
    predict_nme_rank,
    select_exemplar,
)
from ..data.hfmodule import filter_by_classid
from ..metrics.accuracy import accuracy
from ..scheduler import LinearWarmupCosineAnnealingLR

if TYPE_CHECKING:
    import torch.nn as nn
    from datasets import Dataset

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
        per_task_optim_args: dict[int | str, dict] | None = None,
        per_task_sched_args: dict[int | str, dict] | None = None,
        buffer_replay_args: BufferReplayArgs | None = None,
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
        self.task_id: int = None  # ty: ignore[invalid-assignment]
        self.num_old_classes: int = None  # ty: ignore[invalid-assignment]
        self.num_seen_classes: int = None  # ty: ignore[invalid-assignment]

        self.data_column_translate: dict[str, str] = data_column_translate or {}
        # kwargs for optimizer/scheduler per task_id
        # e.g. {0: {"type":"sgd", "lr":0.1}, 1: {"type":"sgd", "lr":0.01}}
        # first task SGD(lr=0.1), second task SGD(lr=0.01)
        self.per_task_optim_args: dict[int | str, dict] = per_task_optim_args or {}
        self.per_task_sched_args: dict[int | str, dict] = per_task_sched_args or {}

        self.buffer_replay_args: BufferReplayArgs = (
            buffer_replay_args or BufferReplayArgs()
        )

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
            # in sync, no update needed
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
            # task id increased, but no new classes, likely manually set var bugs
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

        # a waterfall lookup for optimizer/scheduler configs:
        # per-task specific -> default -> {}
        optim_kwargs = (
            self.per_task_optim_args.get(self.task_id)
            or self.per_task_optim_args.get("default")
            or {}
        )
        sched_kwargs = (
            self.per_task_sched_args.get(self.task_id)
            or self.per_task_sched_args.get("default")
            or {}
        )
        optim = self._get_optimizer(params, **optim_kwargs)
        # If sched_kwargs is None (or explicitly disabled), return optimizer only
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
    def training_step(self, batch, batch_idx: int) -> torch.Tensor: ...

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        dm: HFDataModule = self.trainer.datamodule  # ty: ignore[unresolved-attribute]
        name = getattr(dm, "_val_loader_names", None)
        suffix = (
            name[dataloader_idx] if name is not None else f"dataloader{dataloader_idx}"
        )

        x, y = self.unpack_batch(batch)
        logits: torch.Tensor = self(x)
        acc1 = accuracy(logits, y)
        dict_to_log = {
            f"val-acc1/{suffix}": acc1,
        }

        if self._cached_val_nme is not None:
            topk = self.buffer_replay_args.eval_topk
            class_ids, class_means = self._cached_val_nme
            rank_pred = predict_nme_rank(
                self.feature_extractor, x, class_ids, class_means, topk=topk
            )
            nme_acc1 = rank_pred[:, 0].eq(y).float().mean()

            dict_to_log |= {
                f"val-nmeacc1/{suffix}": nme_acc1,
            }

        self.log_dict(
            dict_to_log,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        dm: HFDataModule = self.trainer.datamodule  # ty: ignore[unresolved-attribute]
        name = getattr(dm, "_test_loader_names", None)
        suffix = (
            name[dataloader_idx] if name is not None else f"dataloader{dataloader_idx}"
        )

        x, y = self.unpack_batch(batch)
        logits: torch.Tensor = self(x)
        acc1 = accuracy(logits, y)
        dict_to_log = {
            f"test-acc1/{suffix}": acc1,
        }

        if self._cached_test_nme is not None:
            topk = self.buffer_replay_args.eval_topk
            class_ids, class_means = self._cached_test_nme
            rank_pred = predict_nme_rank(
                self.feature_extractor, x, class_ids, class_means, topk=topk
            )
            nme_acc1 = rank_pred[:, 0].eq(y).float().mean()

            dict_to_log |= {
                f"test-nmeacc1/{suffix}": nme_acc1,
            }

        self.log_dict(
            dict_to_log,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def on_validation_epoch_start(self) -> None:
        if not self._should_run_val_nme_eval:
            self._cached_val_nme = None
            return

        self._cached_val_nme = self._prepare_buffer_eval(
            self.trainer.datamodule,  # ty: ignore[unresolved-attribute]
        )

    def on_validation_epoch_end(self) -> None:
        self._cached_val_nme = None

    def on_test_epoch_start(self) -> None:
        if not self.buffer_replay_args.eval:
            self._cached_test_nme = None
            return

        self._cached_test_nme = self._prepare_buffer_eval(
            self.trainer.datamodule,  # ty: ignore[unresolved-attribute]
        )

    def on_test_epoch_end(self) -> None:
        self._cached_test_nme = None

    @torch.no_grad()
    def update_memory(self, dm: "HFDataModule") -> None:
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
        self._construct_exemplar(
            dm,
            dm.get_effective_transform_name("test"),
            per_class_quota,
            self.buffer_replay_args.loader_kwargs,
            self.buffer_replay_args.strategy,
        )
        self.train()
        return

    @torch.no_grad()
    def _construct_exemplar(
        self,
        dm: "HFDataModule",
        transform_name: str | None,
        per_class_quota: int,
        loader_kwargs: dict,
        strategy: Literal["random", "herding"] = "herding",
    ) -> None:
        """Construct class exemplars.

        Args:
            dm (HFDataModule):
                Data module to update, containing the buffer.
            transform_name (str | None):
                HuggingFace formatter name to set on the returned dataset.
            per_class_quota (int):
                Number of exemplars to store per class.
            loader_kwargs (dict):
                Additional arguments to DataLoader.
            strategy (Literal["random", "herding"]):
                Exemplar selection strategy. (default: ``"herding"``)

        """
        assert dm.buffer is not None
        old_class_means = {}

        # find means of old classes with newly trained network
        for class_idx in range(self.num_old_classes):
            if f"{class_idx}" not in dm.buffer or len(dm.buffer[f"{class_idx}"]) == 0:
                continue

            loader = dm.buffer.get_dataloader(
                keys=[f"{class_idx}"],
                transform_name=transform_name,
                loader_kwargs=loader_kwargs,
            )
            mean, _ = compute_nme(loader, self.feature_extractor, self.device)
            old_class_means[class_idx] = mean.cpu()

        cur_task_buffer, cur_class_means = self._construct_exemplar_for_cur_task(
            dm, transform_name, per_class_quota, loader_kwargs, strategy=strategy
        )

        dm.buffer.per_class_means = old_class_means | cur_class_means
        dm.buffer.update(cur_task_buffer)
        return

    def _construct_exemplar_for_cur_task(
        self,
        dm: "HFDataModule",
        transform_name: str | None,
        per_class_quota: int,
        loader_kwargs: dict,
        strategy: Literal["random", "herding"] = "herding",
    ) -> "tuple[dict[str, Dataset], dict[int, torch.Tensor]]":
        """Construct exemplars for current task' classes.

        Args:
            dm (HFDataModule):
                Data module to update, containing the buffer.
            transform_name (str | None):
                HuggingFace formatter name to set on the returned dataset.
            per_class_quota (int):
                Number of exemplars to store per class.
            loader_kwargs (dict):
                Additional arguments to DataLoader.
            strategy (Literal["random", "herding"]):
                Exemplar selection strategy. (default: ``"herding"``)

        """
        # narrow down dataset to include only current classes to boost filtering
        cur_task_subset = dm.get_filtered_dataset(
            split=dm._split_train,
            filter_fn=partial(
                filter_by_classid, _min=self.num_old_classes, _max=self.num_seen_classes
            ),
            transform_name=transform_name,
            use_buffer=False,
        )
        class_subset_lookup: dict[int, Dataset] = {
            class_idx: cur_task_subset.filter(
                partial(filter_by_classid, _min=class_idx, _max=class_idx + 1)
            )
            for class_idx in range(self.num_old_classes, self.num_seen_classes)
        }

        ret_buffer: dict[str, Dataset] = {}
        per_class_means: dict[int, torch.Tensor] = {}

        # construct exemplar set for current classes
        for class_idx, class_subset in class_subset_lookup.items():
            n_samples = len(class_subset)
            if n_samples == 0:
                raise RuntimeError(
                    f"No samples found for class {class_idx} in current task dataset."
                )

            # select exemplars by herding
            selected_idx = select_exemplar(
                size=per_class_quota,
                dataset_size=n_samples,
                dataloader=torch.utils.data.DataLoader(
                    class_subset,  # ty: ignore[invalid-argument-type]
                    **loader_kwargs,
                ),
                feature_extractor=self.feature_extractor,
                device=self.device,
                strategy=strategy,
                seed_offset=class_idx,
            )
            selected_dataset = class_subset.select(selected_idx)

            # recompute class mean after selection
            loader = torch.utils.data.DataLoader(
                selected_dataset,
                **loader_kwargs,
            )
            mean, _ = compute_nme(loader, self.feature_extractor, self.device)

            # reset format and store in buffer/class_mean
            selected_dataset.reset_format()
            ret_buffer[f"{class_idx}"] = selected_dataset
            per_class_means[class_idx] = mean.cpu()

        return ret_buffer, per_class_means

    @torch.no_grad()
    def _prepare_buffer_eval(
        self,
        dm: "HFDataModule",
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if dm.buffer is None:
            return None

        transform_name = dm.get_effective_transform_name("test")
        per_class_means: dict[int, torch.Tensor] = {}

        # find means of old classes with newly trained network
        for class_idx in range(self.num_old_classes):
            key = f"{class_idx}"
            if key not in dm.buffer or len(dm.buffer[key]) == 0:
                continue

            if self.buffer_replay_args.eval_recompute_old_task:
                loader = dm.buffer.get_dataloader(
                    [key], transform_name, self.buffer_replay_args.loader_kwargs
                )
                class_mean, _ = compute_nme(
                    loader, self.feature_extractor, self.device, normalize_mean=True
                )
                per_class_means[class_idx] = class_mean.cpu()
            elif class_idx in dm.buffer.per_class_means:
                per_class_means[class_idx] = dm.buffer.per_class_means[class_idx].cpu()

        # construct exemplar set for current classes
        if (
            self.buffer_replay_args.eval_compute_cur_task
            and self.num_seen_classes > self.num_old_classes
        ):
            _, cur_class_means = self._construct_exemplar_for_cur_task(
                dm,
                transform_name,
                per_class_quota=dm.buffer.size_per_class(self.num_seen_classes),
                loader_kwargs=self.buffer_replay_args.loader_kwargs,
                strategy=self.buffer_replay_args.strategy,
            )
            per_class_means |= cur_class_means

        if len(per_class_means) == 0:
            return None

        class_ids = sorted(per_class_means.keys())
        class_ids_t = torch.tensor(class_ids, dtype=torch.long)
        class_means_t = torch.stack([per_class_means[c] for c in class_ids], dim=0)
        class_means_t = F.normalize(class_means_t, dim=1)

        return class_ids_t, class_means_t

    @property
    def _should_run_val_nme_eval(self) -> bool:
        # skip if during sanity check
        if self.trainer.sanity_checking:
            return False
        # skip if NME evaluation is disabled explicitly
        if not self.buffer_replay_args.eval:
            return False
        dm: HFDataModule = self.trainer.datamodule  # ty: ignore[unresolved-attribute]
        if not (dm.buffer is not None and len(dm.buffer) > 0):
            return False

        epoch_idx = self.trainer.current_epoch + 1
        max_epochs = self.trainer.max_epochs

        if max_epochs is not None and max_epochs > 0 and epoch_idx >= max_epochs:
            return True

        return epoch_idx % self.buffer_replay_args.eval_every_n_epochs == 0
