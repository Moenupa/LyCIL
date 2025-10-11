import torch

def accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Top-1 accuracy for multi-class logits."""
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean()

def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Top-k accuracy for multi-class logits."""
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(target.view(-1,1)).any(dim=1)
    return correct.float().mean()
