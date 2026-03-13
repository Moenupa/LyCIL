import torch
import torch.nn.functional as F
from .base import BaseLearner


class WA(BaseLearner):
    r"""`WA`_: Maintaining Discrimination and Fairness in Class Incremental Learning.

    Combines weighted classification and distillation during incremental
    training, then applies weight alignment to the newly added classifier
    weights after finishing each non-initial task.

    - First task: standard cross-entropy on all classes.
    - Later tasks: weighted CE + weighted distillation on old classes.
    - End of task: weight alignment + exemplar memory update.

    The incremental loss follows the original WA recipe:

    .. math::
        L = (1 - \lambda) L_\text{CE} + \lambda L_\text{distill},

    where :math:`\lambda = n_\text{old} / n_\text{seen}`.

    Args:
        distill_T (float, optional): Temperature for distillation.
            Default: 2.0.
        args: See :class:`BaseLearner` for other args.
        kwargs: See :class:`BaseLearner` for other args.

    .. _WA:
        https://arxiv.org/abs/1911.07053
    """

    def __init__(self, *args, distill_T: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_T = float(distill_T)

    @property
    def distill_balance(self) -> float:
        r"""Task-dependent mixing weight for distillation.

        - ``0.0`` for the first task.
        - ``n_old / n_seen`` otherwise.
        """
        if self.task_id == 0:
            return 0.0

        return self.num_old_classes / self.num_seen_classes

    @property
    def num_new_classes(self) -> int:
        return self.num_seen_classes - self.num_old_classes

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = self.unpack_batch(batch)
        logits: torch.Tensor = self(x)

        loss_ce = F.cross_entropy(logits, y)
        loss = loss_ce
        loss_distill = None

        if self.task_id > 0:
            old_logits = self.old_self.forward_no_grad(x)
            T = self.distill_T

            p = F.log_softmax(logits[:, : self.num_old_classes] / T, dim=1)
            q = F.softmax(old_logits[:, : self.num_old_classes] / T, dim=1)
            loss_distill = F.kl_div(p, q, reduction="batchmean") * (T * T)

            alpha = self.distill_balance
            loss = (1.0 - alpha) * loss_ce + alpha * loss_distill

        self.log_dict(
            {
                "train/loss": loss,
                "train/ce": loss_ce,
                "train/distill": loss_distill or 0.0,
                "train/distill_balance": self.distill_balance,
                "train/x_mean": x.detach().float().mean(),
                "train/x_var": x.detach().float().var(unbiased=False),
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss


    @torch.no_grad()
    def weight_align(self, num_new_classes: int) -> None:
        r"""Align newly added classifier weights to the old-class weight scale.

        WA rescales the last ``num_new_classes`` classifier rows by

        .. math::
            \gamma = \frac{\mathbb{E}\|w_\text{old}\|_2}
                           {\mathbb{E}\|w_\text{new}\|_2}.

        Args:
            num_new_classes (int): Number of classes introduced in the current
                task. Assumes newly added classifier weights are appended at the
                end of ``self.classifier.weight``.
        """
        if num_new_classes <= 0:
            return

        weight = self.classifier.weight
        num_total_classes = weight.size(0)
        if num_new_classes >= num_total_classes:
            return

        old_weight = weight[:-num_new_classes]
        new_weight = weight[-num_new_classes:]

        old_mean_norm = old_weight.norm(p=2, dim=1).mean()
        new_mean_norm = new_weight.norm(p=2, dim=1).mean()

        eps = torch.finfo(weight.dtype).eps
        gamma = old_mean_norm / new_mean_norm.clamp_min(eps)

        self.classifier.weight[-num_new_classes:].mul_(gamma)


    def on_train_end(self):
        """Align new classifier weights, then update exemplar memory."""
        if self.task_id > 0:
            self.weight_align(self.num_new_classes)
        self.update_memory(self.trainer.datamodule)  # ty: ignore[unresolved-attribute]
