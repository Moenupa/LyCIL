import copy
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal, Optional

import lightning as L
import torch
from torch.optim import lr_scheduler

# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from ..backbone.resnet import ResNetBackbone
from ..classifier import expand_head, make_head
from ..constants import _X_COLUMN_NAME, _Y_COLUMN_NAME
from ..metrics.accuracy import accuracy, accuracy_topk
from ..scheduler import LinearWarmupCosineAnnealingLR

if TYPE_CHECKING:
    import torch.nn as nn

    from ..data.buffer import BaseExemplarBuffer
    from ..data.hfmodule import HFDataModule


class BaseLearner(L.LightningModule):
    r"""Base class providing backbone, head expansion, optimizer, and memory plumbing.

    Subclasses must implement:
      - training_step() with appropriate losses
      - update_memory() to (re)build exemplars for the new classes
      - validation logic (optionally override `validation_step` or `on_validation_epoch_end`)
    """

    def __init__(
        self,
        *,
        backbone_args: dict | None = None,
        head: Literal["linear", "cosine"] = "linear",
        data_column_translate: dict[str, str] | None = None,
        per_task_optim_args: dict[int, dict] | None = None,
        per_task_sched_args: dict[int, dict] | None = None,
    ):
        super().__init__()

        self.backbone = ResNetBackbone(**(backbone_args or {}))
        self.head_type = head
        # lazy init by head_type at `expand_head()`
        self.classifier: Optional["nn.Module"] = None

        self.buffer: Optional["BaseExemplarBuffer"] = None
        self._old_self: Optional["BaseLearner"] = None

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
        return self.backbone.feature_dim

    def set_task_id(self, task_id: int):
        self.task_id = task_id

    def sync_with_datamodule(self, dm: "HFDataModule"):
        """Synchronizes task states with datamodule, including current task ID
        and seen classes.

        Args:
            dm (HFDataModule): Data module to sync with.
        """
        dm_task_id = dm.get_current_task()
        if self.task_id is not None and dm_task_id == self.task_id:
            # in sync, no update
            return
        if self.task_id is not None and self.task_id < 0:
            # a special bypass rule for buffer-only training
            # this will disable head expansion, to do this, you should:
            # set by `learner.task_id=-2` and
            # reset by `learner.task_id=cur_task_id`.
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return batch[_X_COLUMN_NAME], batch[_Y_COLUMN_NAME]

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

    @abstractmethod
    def update_memory(self, *args, **kwargs): ...

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
        # acc5 = accuracy_topk(logits, y, k=min(5, logits.size(1)))
        self.log_dict(
            {
                f"acc1/task{self.task_id}": acc1,
                # f"val/acc5/task{self.task_id}": acc5,
            },
            prog_bar=False,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = self.unpack_batch(batch)
        logits: torch.Tensor = self(x)
        acc1 = accuracy(logits, y)
        # acc5 = accuracy_topk(logits, y, k=min(5, logits.size(1)))
        self.log_dict(
            {
                f"test/acc1/task{self.task_id}": acc1,
                # f"test/acc5/task{self.task_id}": acc5,
            },
            prog_bar=False,
            sync_dist=True,
        )
