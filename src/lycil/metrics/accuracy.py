import torch


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute top-1 classification accuracy.

    Args:
        logits (torch.Tensor): Raw class scores of shape ``(N, C)``.
        target (torch.Tensor): Ground-truth class indices of shape ``(N,)``.

    Returns:
        torch.Tensor: Scalar mean top-1 accuracy in ``[0, 1]``.
    """
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean()


def accuracy_topk(
    logits: torch.Tensor, target: torch.Tensor, k: int = 5
) -> torch.Tensor:
    """Compute top-k classification accuracy.

    Args:
        logits (torch.Tensor): Raw class scores of shape ``(N, C)``.
        target (torch.Tensor): Ground-truth class indices of shape ``(N,)``.
        k (int, optional): Number of top predictions to consider.
            (default: ``5``)

    Returns:
        torch.Tensor: Scalar mean top-k accuracy in ``[0, 1]``.
    """
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(target.view(-1, 1)).any(dim=1)
    return correct.float().mean()
