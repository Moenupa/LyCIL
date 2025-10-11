from typing import List
import torch
import torch.nn.functional as F
from .base import BaseIncremental

class LWF(BaseIncremental):
    """Learning without Forgetting (Li & Hoiem, ECCV 2016).

    - No exemplar memory (mem_size can be 0).
    - Train on current task data only.
    - Cross-entropy on current task labels over all seen classes.
    - Distillation on OLD classes using a frozen previous model.
    """
    def __init__(
        self,
        *args,
        distill_T: float = 2.0,
        lambda_distill: float = 1.0,
        **kwargs
    ):
        # Default to linear head, but allow override
        kwargs.setdefault("head", "linear")
        kwargs.setdefault("mem_size", 0)
        super().__init__(*args, **kwargs)
        self.distill_T = float(distill_T)
        self.lambda_distill = float(lambda_distill)

    def training_step(self, batch, batch_idx):
        x, y = batch  # labels belong to current classes only
        logits = self(x)

        # Standard CE over all seen classes (labels are within current classes)
        loss_ce = F.cross_entropy(logits, y)

        # Distillation on OLD classes only (seen \ current)
        loss_distill = torch.tensor(0.0, device=x.device)
        if self.prev_model is not None and len(self.seen_classes) > 0:
            old_classes = sorted(list(set(self.seen_classes) - set(self.current_classes)))
            if len(old_classes) > 0:
                idx = torch.tensor(old_classes, device=x.device, dtype=torch.long)
                with torch.no_grad():
                    prev_logits = self.prev_model(x)
                T = self.distill_T
                p = F.log_softmax(logits.index_select(1, idx) / T, dim=1)
                q = F.softmax(prev_logits.index_select(1, idx) / T, dim=1)
                loss_distill = F.kl_div(p, q, reduction="batchmean") * (T * T)

        loss = loss_ce + self.lambda_distill * loss_distill
        self.log_dict(
            {"train/loss": loss, "train/ce": loss_ce, "train/distill": loss_distill},
            prog_bar=True, on_epoch=True, sync_dist=True
        )
        return loss

    @torch.no_grad()
    def update_memory(self, *args, **kwargs):
        """LwF stores no exemplars; do nothing."""
        return
