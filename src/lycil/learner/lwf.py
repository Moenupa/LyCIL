import torch
import torch.nn.functional as F

from .base import BaseLearner


class LWF(BaseLearner):
    r"""`Learning without Forgetting`_ (Li & Hoiem, ECCV 2016).

    - Distillation on old classes, CE on new classes.
    - Loss :math:`L = L_\text{CE} + \lambda * L_\text{distill}`.

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
        x, y = self.unpack_batch(batch)
        logits: torch.Tensor = self(x)

        if self.task_id > 0:
            # distill on old classes ($trainset \setminus cur$)
            old_logits = self.old_self.forward_no_grad(x)
            T = self.distill_T

            # mask to only allow old classes in
            p = F.log_softmax(logits[:, : self.num_old_classes] / T, dim=1)
            q = F.softmax(old_logits[:, : self.num_old_classes] / T, dim=1)
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
                "train/loss": loss,
                "train/ce": loss_ce,
                "train/distill": loss_distill or 0.0,
                "train/x_mean": x.detach().float().mean(),
                "train/x_var": x.detach().float().var(unbiased=False),
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss
