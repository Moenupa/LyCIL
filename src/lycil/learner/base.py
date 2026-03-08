import copy
from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Literal

import lightning as L
import torch
from torch.optim import lr_scheduler

from ..backbone import BaseBackbone, ConvNetArgs, ResNetBackbone
from ..classifier import expand_head, make_head
from ..constants import _X_COLUMN_NAME, _Y_COLUMN_NAME
from ..data.buffer import compute_nme
from ..data.hfmodule import filter_by_classid
from ..metrics.accuracy import accuracy, accuracy_topk
from ..scheduler import LinearWarmupCosineAnnealingLR

if TYPE_CHECKING:
    import torch.nn as nn

    from ..data.buffer import BaseExemplarBuffer
    from ..data.hfmodule import HFDataModule


class BaseLearner(L.LightningModule):
    r"""Base class providing backbone, optimizer, and memory helpers.

    Subclasses must implement:
      - ``training_step()`` with appropriate losses
      - ``on_train_end()`` for replaying history tasks, or other strategies.
        By default, memory updates are disabled.

    Args:
        backbone_args (dict, optional):
            Args to init backbone. (default: None)
        head (Literal[linear, cosine], optional):
            Head type. (default: "linear")
        data_column_translate (dict[str, str], optional):
            Data column mapping that translate to Lycil-recogizable column
            names. (default: None)
        per_task_optim_args (dict[int, dict], optional):
            Per-task optimizer arguments. (default: None)
        per_task_sched_args (dict[int, dict], optional):
            Per-task scheduler arguments. (default: None)
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
        self.task_id: int = None  # ty: ignore[invalid-assignment]
        self.num_old_classes: int = None  # ty: ignore[invalid-assignment]
        self.num_seen_classes: int = None  # ty: ignore[invalid-assignment]

        self.data_column_translate: dict[str, str] = data_column_translate or {}
        # kwargs for optimizer/scheduler per task_id
        # e.g. {0: {"type":"sgd", "lr":0.1}, 1: {"type":"sgd", "lr":0.01}}
        # first task SGD(lr=0.1), second task SGD(lr=0.01)
        self.per_task_optim_args: dict[int, dict] = per_task_optim_args or {}
        self.per_task_sched_args: dict[int, dict] = per_task_sched_args or {}

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    def set_task_id(self, task_id: int):
        self.task_id = task_id

    def sync_with_datamodule(self, dm: "HFDataModule"):
        """Synchronizes task states with datamodule.

        - Updates ``task_id`` from datamodule's current task.
        - If datamodule is newer, updates ``num_old_classes`` and ``num_seen_classes``.

        Args:
            dm (HFDataModule): Data module to sync with.
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
        x = batch[_X_COLUMN_NAME]
        y = batch[_Y_COLUMN_NAME]
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        return x, y

    @torch.no_grad()
    def expand_head(self, num_new: int) -> None:
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
        """Forward pass without tracking gradients. Useful for memory updates."""
        return self.forward(x)

    def forward_layerwise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
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
        """Keep a frozen copy of the current model."""
        # prevent recursive copies
        self._old_self = None

        # snapshot and freeze
        self._old_self = copy.deepcopy(self).eval()
        for p in self._old_self.parameters():
            p.requires_grad_(False)

    @property
    def old_self(self) -> "BaseLearner":
        """Returns a frozen copy of the old model. Call `snapshot_old()` to update the snapshot."""
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

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = self.unpack_batch(batch)
        logits: torch.Tensor = self(x)
        acc1 = accuracy(logits, y)
        acc5 = accuracy_topk(logits, y, k=min(5, logits.size(1)))
        self.log_dict(
            {
                f"val/acc1/task{self.task_id}": acc1,
                f"val/acc5/task{self.task_id}": acc5,
            },
            prog_bar=False,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = self.unpack_batch(batch)
        logits: torch.Tensor = self(x)
        acc1 = accuracy(logits, y)
        acc5 = accuracy_topk(logits, y, k=min(5, logits.size(1)))
        self.log_dict(
            {
                f"test/acc1/task{self.task_id}": acc1,
                f"test/acc5/task{self.task_id}": acc5,
            },
            prog_bar=False,
            sync_dist=True,
        )

    @torch.no_grad()
    def update_memory(self, dm: "HFDataModule", **kwargs) -> None:
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

        self.eval()
        if dm.buffer.is_adaptive:
            # vacate exemplars for more classes
            dm.buffer.reduce_exemplars(dm.buffer.size_per_class(self.num_seen_classes))
            self._construct_exemplar(dm, **kwargs)
        else:
            self._construct_exemplar_unified(dm, **kwargs)
        self.train()
        return

    @torch.no_grad()
    def _construct_exemplar(self, dm: "HFDataModule", **kwargs) -> None:
        raise NotImplementedError

        assert dm.buffer is not None
        # construct exemplar set for current classes
        for class_idx in range(self.num_old_classes, self.num_seen_classes):
            pass

    @torch.no_grad()
    def _construct_exemplar_unified(self, dm: "HFDataModule", **kwargs) -> None:
        # for dataloader during exemplar construction,
        # rather conservative because args are hard-coded here
        loader_kwargs = dict(
            batch_size=1,
            shuffle=False,
            num_workers=8,
        )

        assert dm.buffer is not None
        per_class_means = {}

        # find means of old classes with newly trained network
        for class_idx in range(self.num_old_classes):
            loader = dm.buffer.get_dataloader(
                keys=[f"{class_idx}"],
                transform_name=dm.get_effective_transform_name(),
                loader_kwargs=loader_kwargs,
            )
            mean, _ = compute_nme(loader, self.feature_extractor, self.device)
            per_class_means[class_idx] = mean

        # construct exemplar set for current classes
        for class_idx in range(self.num_old_classes, self.num_seen_classes):
            # 1. single pass on all data
            train_loader = dm.get_dataloader(
                split=dm._split_train,
                filter_fn=partial(filter_by_classid, class_idx=class_idx),
                transform_name=dm.get_effective_transform_name(),
                loader_kwargs=loader_kwargs,
            )
            mean, per_sample_features = compute_nme(
                train_loader, self.feature_extractor, self.device
            )

            # 2. select exemplars by herding
            # for now, use first m samples
            m = dm.buffer.size_per_class(self.num_seen_classes)
            selected_idx = list(range(0, m))
            # TODO: implement full herding
            # herding implementation from another library is below:
            # selected_exemplars = []
            # exemplar_vectors = []
            # for k in range(1, m + 1):
            #     S = np.sum(
            #         exemplar_vectors, axis=0
            #     )  # [feature_dim] sum of selected exemplars vectors
            #     mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
            #     i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

            #     selected_exemplars.append(
            #         np.array(data[i])
            #     )  # New object to avoid passing by inference
            #     exemplar_vectors.append(
            #         np.array(vectors[i])
            #     )  # New object to avoid passing by inference

            #     vectors = np.delete(
            #         vectors, i, axis=0
            #     )  # Remove it to avoid duplicative selection
            #     data = np.delete(
            #         data, i, axis=0
            #     )  # Remove it to avoid duplicative selection
            selected_dataset = dm.get_filtered_dataset(
                split=dm._split_train,
                filter_fn=partial(filter_by_classid, class_idx=class_idx),
            ).select(selected_idx)
            dm.buffer[f"{class_idx}"] = selected_dataset

            # 3. recompute class mean after selection
            loader = dm.buffer.get_dataloader(
                keys=[f"{class_idx}"],
                transform_name=dm.get_effective_transform_name(),
                loader_kwargs=loader_kwargs,
            )
            mean, _ = compute_nme(loader, self.feature_extractor, self.device)
            per_class_means[class_idx] = mean

        dm.buffer.per_class_means = per_class_means
