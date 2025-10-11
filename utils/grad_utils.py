from typing import Iterable, List, Tuple
import torch
import itertools

def iter_params(model: torch.nn.Module):
    """Yield parameters with requires_grad=True in a stable order."""
    for p in model.parameters():
        if p.requires_grad:
            yield p

def pack_grads(model: torch.nn.Module) -> torch.Tensor:
    """Flatten current parameter gradients into a single vector."""
    grads = []
    for p in iter_params(model):
        if p.grad is None:
            grads.append(torch.zeros_like(p).flatten())
        else:
            grads.append(p.grad.detach().flatten())
    return torch.cat(grads) if grads else torch.tensor([], device=next(model.parameters()).device)

def zero_grads(model: torch.nn.Module):
    for p in iter_params(model):
        if p.grad is not None:
            p.grad.zero_()

def apply_grad_vector(model: torch.nn.Module, g: torch.Tensor):
    """Set parameter .grad from a flat vector g (same layout as pack_grads)."""
    offset = 0
    for p in iter_params(model):
        n = p.numel()
        p.grad = g[offset:offset+n].view_as(p).clone()
        offset += n
    assert offset == g.numel()
