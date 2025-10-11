from typing import Dict, List
import torch
import torch.nn.functional as F

def _spatial_pool_pyramid(f: torch.Tensor, levels: List[int]) -> torch.Tensor:
    """Return concatenated pooled vectors over pyramid levels (e.g., [1,2,4])."""
    outs = []
    B, C, H, W = f.shape
    for L in levels:
        if L == 1:
            v = f.mean(dim=[2,3])
        else:
            # average pooling to LxL, then flatten
            v = F.adaptive_avg_pool2d(f, (L, L)).reshape(B, C*L*L)
        outs.append(v)
    return torch.cat(outs, dim=1)

def pod_loss(curr: Dict[str, torch.Tensor], prev: Dict[str, torch.Tensor], levels: List[int] = [1,2,4]) -> torch.Tensor:
    """Pooled Output Distillation loss across multiple layers.
    Normalize pooled descriptors and compute L2 difference.
    """
    loss = 0.0
    count = 0
    for k in curr.keys():
        if k not in prev:
            continue
        c = _spatial_pool_pyramid(curr[k], levels)
        p = _spatial_pool_pyramid(prev[k], levels)
        # normalize
        c = F.normalize(c, dim=1)
        p = F.normalize(p, dim=1)
        loss = loss + torch.mean((c - p) ** 2)
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=list(curr.values())[0].device)
    return loss / count
