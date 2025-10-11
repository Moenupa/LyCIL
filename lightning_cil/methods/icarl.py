from typing import List
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from .base import BaseIncremental
from torch.utils.data import ConcatDataset, DataLoader

class ICaRL(BaseIncremental):
    """iCaRL: Incremental Classifier and Representation Learning.

    - Sigmoid BCE classification over all seen classes
    - Distillation: BCE between current and previous logits on old classes
    - Exemplar memory: herding + NME-based evaluation (optional)
    """
    def __init__(self, *args, bce_targets_smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bce_targets_smoothing = float(bce_targets_smoothing)

    def training_step(self, batch, batch_idx):
        x, y = batch  # y are global class ids
        logits = self(x)  # (B, C_seen)
        B, C = logits.size()
        C_seen = len(self.seen_classes)

        # Build multi-label targets (B, C), with label smoothing if set
        y_onehot = torch.zeros((B, C), device=logits.device)
        y_onehot.scatter_(1, y.view(-1,1), 1.0)
        if self.bce_targets_smoothing > 0:
            y_onehot = y_onehot * (1 - self.bce_targets_smoothing) + self.bce_targets_smoothing / C

        loss_cls = F.binary_cross_entropy_with_logits(logits, y_onehot)

        # Distillation for previously seen classes against prev_model
        loss_dist = torch.tensor(0.0, device=logits.device)
        if self.prev_model is not None and C_seen > 0:
            with torch.no_grad():
                prev_logits = self.prev_model(x)
            old_idx = torch.tensor(sorted(self.seen_classes), device=logits.device, dtype=torch.long)
            old_idx = old_idx[old_idx < C]  # safety for early tasks
            if old_idx.numel() > 0:
                loss_dist = F.binary_cross_entropy_with_logits(
                    logits.index_select(1, old_idx),
                    torch.sigmoid(prev_logits.index_select(1, old_idx))
                )

        loss = loss_cls + loss_dist
        self.log_dict({"train/loss": loss, "train/loss_cls": loss_cls, "train/loss_dist": loss_dist},
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def update_memory(self, datamodule, device: torch.device):
        """Apply herding per new class and rebalance the buffer quota."""
        # Reduce existing exemplars
        per_class = self.buffer.exemplars_per_class(num_classes_seen=len(self.seen_classes))
        self.buffer.reduce_exemplars(per_class)

        # Build exemplars for each new class
        for c in self.current_classes:
            loader = datamodule.build_class_loader(c, train=True, batch_size=128, shuffle=False, num_workers=datamodule.num_workers)
            self.buffer.build_exemplars_herding(
                model=self, feature_extractor=self.feature_extractor,
                dataloader=loader, class_id=c, m=per_class, device=device
            )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.use_nme_eval and len(self.buffer) > 0:
            logits = self.predict_nme(x)
        else:
            logits = self(x)
        acc1 = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/acc1", acc1, prog_bar=True, on_epoch=True, sync_dist=True)
