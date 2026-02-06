import torch
import torch.nn.functional as F

from ..constants import _X_COLUMN_NAME, _Y_COLUMN_NAME
from .base import BaseLearner


class LWF(BaseLearner):
    r"""`Learning without Forgetting`_ (Li & Hoiem, ECCV 2016).
    Loss :math:`L = L_\text{CE} + \lambda * L_\text{distill}`.

    Args:
        distill_T (float, optional): Temperature for distillation. Default: 2.0.
        lambda_distill (float, optional): Weight for distillation loss. Default: 1.0.
        args: See :class:`BaseLearner` for other args.
        kwargs: See :class:`BaseLearner` for other args.

    .. _Learning without Forgetting:
        https://arxiv.org/abs/1606.09282
    """

    def __init__(
        self, *args, distill_T: float = 2.0, distill_lambda: float = 1.0, **kwargs
    ):
        # Default to linear head, but allow override
        super().__init__(*args, **kwargs)

        self.distill_T = float(distill_T)
        self.distill_lambda = float(distill_lambda)


    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # labels belong to current classes only
        x = batch[_X_COLUMN_NAME]
        y = batch[_Y_COLUMN_NAME]
        logits: torch.Tensor = self(x)

        if self.prev_model is not None and self.task_id > 0:
            # distill on old classes ($trainset \setminus cur$)
            prev_logits = self.forward_prev(x)
            T = self.distill_T

            # mask to only allow old classes in
            p = F.log_softmax(logits[:, : self.num_old_classes] / T, dim=1)
            q = F.softmax(prev_logits[:, : self.num_old_classes] / T, dim=1)
            loss_distill = F.kl_div(p, q, reduction="batchmean") * (T * T)

            # ce on current classes
            loss_ce = F.cross_entropy(
                logits[:, self.num_old_classes :], y - self.num_old_classes
            )
            loss = loss_ce + self.distill_lambda * loss_distill
        else:
            # first task, no distill
            loss_distill = None
            loss_ce = F.cross_entropy(logits, y)
            loss = loss_ce

        self.log_dict(
            {
                f"train/task{self.task_id}/loss": loss,
                f"train/task{self.task_id}/ce": loss_ce,
                f"train/task{self.task_id}/distill": loss_distill or 0.0,
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
