import torch
import torch.nn as nn
import torch.nn.functional as F
from .replay import Replay

class WA(Replay):
    """Weight Aligning (WA) post-calibration after each task.

    Align weight norms between old vs new classes for a linear head.
    """
    def __init__(self, *args, wa_eps: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.wa_eps = float(wa_eps)

    @torch.no_grad()
    def align_weights(self):
        head = self.classifier
        if not isinstance(head, nn.Linear):
            return  # cosine head already normalized
        if head.out_features == 0 or len(self.current_classes) == 0:
            return
        W = head.weight.data  # (C, D)
        old_idx = [c for c in self.seen_classes if c not in self.current_classes]
        new_idx = self.current_classes
        if len(old_idx) == 0 or len(new_idx) == 0:
            return
        old_norm = W[old_idx].norm(dim=1).mean()
        new_norm = W[new_idx].norm(dim=1).mean() + self.wa_eps
        ratio = (old_norm / new_norm).clamp(min=0.0, max=10.0)
        W[new_idx] = W[new_idx] * ratio

    @torch.no_grad()
    def update_memory(self, datamodule, device: torch.device):
        super().update_memory(datamodule, device)
        self.align_weights()
