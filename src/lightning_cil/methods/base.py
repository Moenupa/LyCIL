import copy
from abc import abstractmethod
from typing import Literal, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.resnet import ResNetBackbone
from ..classifier.cosine_classifier import CosineClassifier
from ..data.buffer import ExemplarBuffer
from ..metrics.accuracy import accuracy, accuracy_topk

_CLASSIFIER_HEADS = {
    # key: (class, {optional kwargs})
    "linear": (nn.Linear, {}),
    "cosine": (CosineClassifier, {"learn_scale": True}),
}


class BaseIncremental(pl.LightningModule):
    """Base class providing backbone, head expansion, optimizer, and memory plumbing.

    Subclasses must implement:
      - training_step() with appropriate losses
      - update_memory() to (re)build exemplars for the new classes
      - validation logic (optionally override `validation_step` or `on_validation_epoch_end`)
    """

    def __init__(
        self,
        backbone_name: Literal["resnet18", "resnet34", "resnet50"] = "resnet18",
        head: Literal["linear", "cosine"] = "linear",
        pretrained_backbone: bool = False,
        lr: float = 0.1,
        weight_decay: float = 1e-4,
        optimizer: Literal["sgd", "adamw"] = "sgd",
        momentum: float = 0.9,
        nesterov: bool = True,
        mem_size: int = 2000,
        use_nme_eval: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["buffer"])
        self.backbone = ResNetBackbone(backbone_name, pretrained_backbone)
        self.head_type = head

        # Delay classifier creation until first task
        self.classifier: Optional[nn.Module] = None

        self.buffer = ExemplarBuffer(mem_size=mem_size)
        self.prev_model: Optional[nn.Module] = None  # frozen copy for distillation
        self.current_classes: list[int] = []
        self.seen_classes: list[int] = []
        self.use_nme_eval = bool(use_nme_eval)

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    # ----- task orchestration ------
    def set_task_info(self, current_classes: list[int], seen_classes: list[int]):
        self.current_classes = list(current_classes)
        self.seen_classes = list(seen_classes)

    def _make_head(self, out_features: int) -> nn.Module:
        """Create a new classification head."""
        assert self.head_type in _CLASSIFIER_HEADS, (
            f"Unknown head type: {self.head_type} not in {_CLASSIFIER_HEADS.keys()}"
        )

        cls, kwargs = _CLASSIFIER_HEADS[self.head_type]
        return cls(self.feature_dim, out_features, **kwargs)

    def expand_head(self, num_new: int):
        """Expand classifier to accommodate `num_new` new classes."""
        assert num_new >= 0, f"num_new {num_new} not >= 0"
        if num_new == 0:
            return
        if self.classifier is None:
            self.classifier = self._make_head(num_new)
            return

        if isinstance(self.classifier, nn.Linear):
            old_out = self.classifier.out_features
            new_linear = nn.Linear(self.feature_dim, old_out + num_new, bias=True)
            with torch.no_grad():
                new_linear.weight[:old_out] = self.classifier.weight.data
                new_linear.bias[:old_out] = self.classifier.bias.data
            self.classifier = new_linear
            return

        if isinstance(self.classifier, CosineClassifier):
            self.classifier.expand(num_new)
            return

        raise RuntimeError("Unknown head type")

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_extractor(x)
        if self.classifier is None:
            raise RuntimeError(
                "Classifier head is not initialized. Call expand_head before training."
            )
        logits = self.classifier(f)
        return logits

    @abstractmethod
    def update_memory(self, *args, **kwargs): ...

    # ----- Lightning hooks -----
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if self.hparams.optimizer.lower() == "sgd":
            opt = torch.optim.SGD(
                params,
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                nesterov=self.hparams.nesterov,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer.lower() == "adamw":
            opt = torch.optim.AdamW(
                params,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError("Unsupported optimizer")
        # Cosine annealing commonly used in CIL
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    @torch.no_grad()
    def snapshot_prev_model(self):
        """
        Keep a frozen copy for distillation across the next task.
        """
        self.prev_model = copy.deepcopy(self).eval()
        for p in self.prev_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def calc_logits_prev(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass with `self.prev_model` on `x`.
        """
        assert self.prev_model is not None, "No previous model stored"
        return self.prev_model(x)

    @abstractmethod
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor: ...

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits: torch.Tensor = self(x)
        acc1 = accuracy(logits, y)
        k = min(5, logits.size(1))
        acc5 = accuracy_topk(logits, y, k=k)
        self.log("val/acc1", acc1, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"val/acc{k}", acc5, prog_bar=False, on_epoch=True, sync_dist=True)

    # ----- NME evaluation for iCaRL -----
    @torch.no_grad()
    def predict_nme(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits built from NME distances to exemplar means."""
        device = x.device
        nme = self.buffer.compute_nme_classifier(self, self.feature_extractor, device)
        if not nme:
            return self(x)
        class_ids = sorted(nme.keys())
        means = torch.stack([nme[c] for c in class_ids]).to(device)  # (C,d)
        f = self.feature_extractor(x)
        f = F.normalize(f, dim=1)
        means = F.normalize(means, dim=1)
        logits = f @ means.t()
        # map to full class space
        out_dim = (
            self.classifier.weight.size(0)
            if hasattr(self.classifier, "weight")
            else max(class_ids) + 1
        )
        full = x.new_zeros((x.size(0), out_dim), device=device)
        full[:, class_ids] = logits
        return full
