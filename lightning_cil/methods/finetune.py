import torch
import torch.nn.functional as F
from .base import BaseIncremental

class FineTune(BaseIncremental):
    """Baseline: train on current task only, no memory, no regularization."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mem_size", 0)
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def update_memory(self, *args, **kwargs):
        # No exemplar memory
        return
