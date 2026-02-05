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


class BaseLearner(L.LightningModule):
    r"""Base class providing backbone, head expansion, optimizer, and memory plumbing.

    Subclasses must implement:
      - training_step() with appropriate losses
      - update_memory() to (re)build exemplars for the new classes
      - validation logic (optionally override `validation_step` or `on_validation_epoch_end`)
    """

    def __init__(
        self,
        num_classes_per_task: int,
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
        self.prev_model: Optional["nn.Module"] = None  # frozen copy for distillation

        # lazy init by `set_task_id()` to sync with data module
        self.task_id: int = None
        self.num_classes_per_task: int = num_classes_per_task

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

    @property
    def num_old_classes(self) -> int:
        """Total number of classes :math:`all \setminus current`,
        equal to the 0-index offset for current class.
        - :math:`\le` this offset means old classes;
        - :math:`\geq` this offset means current new classes.

        Raises:
            RuntimeError: If task_id is not set.

        Returns:
            int: Current class ID offset.
        """
        if self.task_id is None:
            raise RuntimeError("task_id is not set. Call `set_task_id()` first.")
        return self.task_id * self.num_classes_per_task

    @property
    def num_seen_classes(self) -> int:
        """Total number of classes seen so far.

        Raises:
            RuntimeError: If task_id is not set.

        Returns:
            int: Number of seen classes.
        """
        if self.task_id is None:
            raise RuntimeError("task_id is not set. Call `set_task_id()` first.")
        return (self.task_id + 1) * self.num_classes_per_task

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
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError(
                "Classifier head is not initialized. Call expand_head before training."
            )

        f = self.feature_extractor(x)
        logits = self.classifier(f)
        return logits

    @abstractmethod
    def update_memory(self, *args, **kwargs): ...

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]

        optim_kwargs = (
            self.per_task_optim_args.get(self.task_id)
            or self.per_task_optim_args.get(-1)
            or {}
        )
        opt_type = optim_kwargs.pop("type", "sgd")
        match opt_type:
            case "sgd":
                optim = torch.optim.SGD(params, **optim_kwargs)
            case "adamw":
                optim = torch.optim.AdamW(params, **optim_kwargs)
            case _:
                raise NotImplementedError(f"Unsupported optimizer: `{opt_type}`")

        sched_kwargs = (
            self.per_task_sched_args.get(self.task_id)
            or self.per_task_sched_args.get(-1)
            or {}
        )
        sched_type = sched_kwargs.pop("type", "linear_warmup_cosine_annealing")
        match sched_type:
            case "linear_warmup_cosine_annealing":
                sched = LinearWarmupCosineAnnealingLR(optim, **sched_kwargs)
            case "cosine_annealing":
                sched = lr_scheduler.CosineAnnealingLR(optim, **sched_kwargs)
            case "step_lr":
                sched = lr_scheduler.StepLR(optim, **sched_kwargs)
            case "multi_step_lr":
                sched = lr_scheduler.MultiStepLR(optim, **sched_kwargs)
            case _:
                raise NotImplementedError(f"Unsupported scheduler: `{sched_type}`")

        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    @torch.no_grad()
    def snapshot_prev(self):
        """Keep a frozen copy of the current model."""
        # prevent recursive copies
        self.prev_model = None

        # snapshot and freeze
        self.prev_model = copy.deepcopy(self).eval()
        for p in self.prev_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward_prev(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass on `x` with previous snapshot of the model.

        Raises:
            RuntimeError: If previous snapshot is not available.

        Returns:
            torch.Tensor: ``prev_model(x)``
        """
        if self.prev_model is None:
            raise RuntimeError(
                "No previous model stored. Call `snapshot_prev()` first."
            )
        return self.prev_model(x)

    @abstractmethod
    def training_step(self, batch, batch_idx: int) -> torch.Tensor: ...

    def validation_step(self, batch, batch_idx: int) -> None:
        if isinstance(batch, (tuple, list)):
            x, y = batch
        else:
            x = batch[_X_COLUMN_NAME]
            y = batch[_Y_COLUMN_NAME]
        logits: torch.Tensor = self(x)
        acc1 = accuracy(logits, y)
        acc5 = accuracy_topk(logits, y, k=5)
        self.log_dict(
            {
                f"val/task{self.task_id}/acc1": acc1,
                f"val/task{self.task_id}/acc5": acc5,
            },
            prog_bar=False,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx: int) -> None:
        if isinstance(batch, (tuple, list)):
            x, y = batch
        else:
            x = batch[_X_COLUMN_NAME]
            y = batch[_Y_COLUMN_NAME]
        logits: torch.Tensor = self(x)
        acc1 = accuracy(logits, y)
        acc5 = accuracy_topk(logits, y, k=5)
        self.log_dict(
            {
                f"test/task{self.task_id}/acc1": acc1,
                f"test/task{self.task_id}/acc5": acc5,
            },
            prog_bar=False,
            sync_dist=True,
        )
