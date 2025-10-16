import torch
import torch.nn.functional as F

from .base import BaseIncremental


class LWF(BaseIncremental):
    """
    `Learning without Forgetting`_ (Li & Hoiem, ECCV 2016).
    - Train on current task data only.
    - Cross-entropy on current task labels over all seen classes.
    - Distillation on OLD classes using a frozen previous model.
    - Loss :math:`L = L_\text{CE} + \lambda * L_\text{distill}`

    Args:
        distill_T (float, optional): Temperature for distillation. Default: 2.0.
        lambda_distill (float, optional): Weight for distillation loss. Default: 1.0.
        args: See :class:`BaseIncremental` for other args.
        kwargs: See :class:`BaseIncremental` for other args.

    .. _Learning without Forgetting:
        https://arxiv.org/abs/1606.09282
    """

    def __init__(
        self, *args, distill_T: float = 2.0, lambda_distill: float = 1.0, **kwargs
    ):
        # Default to linear head, but allow override
        kwargs.setdefault("head", "linear")
        kwargs.setdefault("mem_size", 0)
        super().__init__(*args, **kwargs)

        self.distill_T = float(distill_T)
        self.lambda_distill = float(lambda_distill)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # labels belong to current classes only
        x, y = batch
        logits: torch.Tensor = self(x)

        # Standard CE over all seen classes (labels are within current classes)
        assert logits.size(0) == y.size(0)
        loss_ce = F.cross_entropy(logits, y)

        # Distillation on OLD classes only (seen \ current)
        loss_distill = torch.tensor(0.0, device=x.device)
        if self.prev_model is not None and len(self.seen_classes) > 0:
            old_classes = list(set(self.seen_classes) - set(self.current_classes))
            if len(old_classes) > 0:
                idx = torch.tensor(
                    sorted(old_classes), device=x.device, dtype=torch.long
                )
                prev_logits = self.calc_logits_prev(x)
                T = self.distill_T
                p = F.log_softmax(logits.index_select(1, idx) / T, dim=1)
                q = F.softmax(prev_logits.index_select(1, idx) / T, dim=1)
                loss_distill = F.kl_div(p, q, reduction="batchmean") * (T * T)

        # combined loss
        loss = loss_ce + self.lambda_distill * loss_distill
        self.log_dict(
            {
                "train/loss": loss,
                "train/ce": loss_ce,
                "train/distill": loss_distill,
            },
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    @torch.no_grad()
    def update_memory(self, *args, **kwargs):
        """LwF stores no exemplars; do nothing."""
        return
