import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineClassifier(nn.Module):
    """Cosine classifier with normalized weights and features.

    Supports dynamic expansion via `expand(num_new)`.
    """
    def __init__(self, in_features: int, out_features: int, learn_scale: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) if False else nn.init.normal_(self.weight, 0, 0.01)
        self.s = nn.Parameter(torch.tensor(10.0)) if learn_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(x, dim=1)
        logits = x @ w.t()
        if self.s is not None:
            logits = logits * self.s
        return logits

    def expand(self, num_new: int):
        if num_new <= 0:
            return
        old_w = self.weight.data
        new_w = torch.randn(old_w.size(1) * 0 + num_new, old_w.size(1), device=old_w.device)
        nn.init.normal_(new_w, 0, 0.01)
        self.weight = torch.nn.Parameter(torch.cat([old_w, new_w], dim=0))

import math
