from typing import List, Dict, Optional, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import ConcatDataset, DataLoader
from models.backbone.resnet import ResNetBackbone
from models.classifier.cosine_classifier import CosineClassifier
from data.buffer import ExemplarBuffer
from utils.metrics import accuracy, accuracy_topk

class BaseIncremental(pl.LightningModule):
    """Base class providing backbone, head expansion, optimizer, and memory plumbing.

    Subclasses must implement:
      - training_step() with appropriate losses
      - update_memory() to (re)build exemplars for the new classes
      - validation logic (optionally override `validation_step` or `on_validation_epoch_end`)
    """
    def __init__(
        self,
        num_classes_total: int,
        backbone_name: str = "resnet18",
        head: str = "linear",           # "linear" or "cosine"
        pretrained_backbone: bool = False,
        lr: float = 0.1,
        weight_decay: float = 1e-4,
        optimizer: str = "sgd",
        momentum: float = 0.9,
        nesterov: bool = True,
        mem_size: int = 2000,
        use_nme_eval: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["buffer"])
        self.num_classes_total = int(num_classes_total)
        self.backbone = ResNetBackbone(backbone_name, pretrained_backbone)
        self.feature_dim = self.backbone.feature_dim
        self.head_type = head

        if head == "linear":
            self.classifier = nn.Linear(self.feature_dim, 0, bias=True)  # init as empty; expand on task 0
        elif head == "cosine":
            self.classifier = CosineClassifier(self.feature_dim, 0, learn_scale=True)
        else:
            raise ValueError("Unsupported head")

        self.buffer = ExemplarBuffer(mem_size=mem_size)
        self.prev_model: Optional[nn.Module] = None  # frozen copy for distillation
        self.seen_classes: List[int] = []
        self.current_classes: List[int] = []
        self.use_nme_eval = bool(use_nme_eval)

    # ----- task orchestration ------
    def set_task_info(self, current_classes: List[int], seen_classes: List[int]):
        self.current_classes = list(current_classes)
        self.seen_classes = list(seen_classes)

    def expand_head(self, num_new: int):
        """Expand classifier to accommodate `num_new` new classes."""
        if num_new <= 0:
            return
        if isinstance(self.classifier, nn.Linear):
            old_out = self.classifier.out_features
            new_linear = nn.Linear(self.feature_dim, old_out + num_new, bias=True)
            # copy old weights if exist
            if old_out > 0:
                with torch.no_grad():
                    new_linear.weight[:old_out] = self.classifier.weight.data
                    new_linear.bias[:old_out] = self.classifier.bias.data
            self.classifier = new_linear
        elif isinstance(self.classifier, CosineClassifier):
            self.classifier.expand(num_new)
        else:
            raise RuntimeError("Unknown head type")

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_extractor(x)
        logits = self.classifier(f)
        return logits

    # ----- memory API to override -----
    def update_memory(self, *args, **kwargs):
        raise NotImplementedError

    # ----- Lightning hooks -----
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if self.hparams.optimizer.lower() == "sgd":
            opt = torch.optim.SGD(
                params, lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                nesterov=self.hparams.nesterov,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer.lower() == "adamw":
            opt = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError("Unsupported optimizer")
        # Cosine annealing commonly used in CIL
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval":"epoch"}}

    # ----- helpers -----
    @torch.no_grad()
    def snapshot_prev_model(self):
        """Keep a frozen copy for distillation across the next task."""
        self.prev_model = copy.deepcopy(self).eval()
        for p in self.prev_model.parameters():
            p.requires_grad = False

    def logits_prev(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.prev_model is None:
            return None
        return self.prev_model(x)

    def training_step(self, batch, batch_idx):
        """Must be implemented in subclasses (iCaRL / LUCIR)."""
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc1 = accuracy(logits, y)
        acc5 = accuracy_topk(logits, y, k=min(5, logits.size(1)))
        self.log("val/acc1", acc1, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/acc5", acc5, prog_bar=False, on_epoch=True, sync_dist=True)

    # ----- NME evaluation for iCaRL -----
    @torch.no_grad()
    def predict_nme(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits built from NME distances to exemplar means."""
        device = x.device
        # build nme table on the fly
        nme = self.buffer.compute_nme_classifier(self, self.feature_extractor, device)
        if not nme:
            return self(x)
        class_ids = sorted(nme.keys())
        means = torch.stack([nme[c] for c in class_ids]).to(device)  # (C,d)
        f = self.feature_extractor(x)
        f = F.normalize(f, dim=1)
        means = F.normalize(means, dim=1)
        # cosine similarity as logits
        logits = f @ means.t()
        # map to full class space
        full = x.new_zeros((x.size(0), self.classifier.weight.size(0))) if hasattr(self.classifier, "weight") else x.new_zeros((x.size(0), 0))
        full = full.to(x.device)
        full[:, class_ids] = logits
        return full
