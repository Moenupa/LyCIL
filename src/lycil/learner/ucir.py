import math

import torch
import torch.nn.functional as F

from .base import BaseLearner


class UCIR(BaseLearner):
    r"""`UCIR`_: Learning a Unified Classifier Incrementally via Rebalancing. (Hou et al., CVPR 2019).

    Requires a cosine classifier head (``head="cosine"``). Combines three loss
    components per task:

    - **Cross-entropy** on all seen classes.
    - **Less-forgetting loss**: cosine embedding loss between current and
      old-model feature vectors, weighted by :attr:`task_factor`.
    - **Inter-class separation loss**: margin ranking loss pushing top-K new
      class scores below the ground-truth old-class score, applied only to
      samples from old classes.

    Args:
        lambda_lf (float, optional): Weight for the less-forgetting cosine
            embedding loss. (default: ``10.0``)
        K (int, optional): Number of hard negatives from new classes used in
            the inter-class separation loss. (default: ``2``)
        margin (float, optional): Margin for
            :func:`~torch.nn.functional.margin_ranking_loss`.
            (default: ``0.5``)
        args: See :class:`BaseLearner` for additional positional arguments.
        kwargs: See :class:`BaseLearner` for additional keyword arguments.
            Must include ``head="cosine"``.

    Raises:
        ValueError: If ``head`` is not set to ``"cosine"``.

    .. _UCIR:
        http://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html
    """

    def __init__(
        self,
        *args,
        lambda_lf: float = 10.0,
        K: int = 2,
        margin: float = 0.5,
        **kwargs,
    ):
        if kwargs.get("head", "linear") != "cosine":
            raise ValueError(f"{self.__class__.__name__} requires head='cosine'.")

        super().__init__(*args, **kwargs)

        self.lambda_lf = float(lambda_lf)
        self.K = int(K)
        self.margin = float(margin)

    @property
    def task_factor(self) -> float:
        r"""Task-dependent scale for the less-forgetting loss.

        - ``0.0`` for the first task (no forgetting to guard against), otherwise
        - Computed as :math:`\sqrt{n_{seen} / n_{new}}` where :math:`n_{new}`
        is the number of classes added in the current task.
        """
        if self.task_id == 0:
            return 0.0

        return math.sqrt(
            self.num_seen_classes / (self.num_seen_classes - self.num_old_classes)
        )

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = self.unpack_batch(batch)
        outputs = self.forward_layerwise(x)
        logits, features = outputs["logits"], outputs["features"]

        loss_ce = F.cross_entropy(logits, y)
        loss = loss_ce

        loss_less_forgetting = None
        loss_interclass_separation = None

        if self.task_id > 0:
            # 1. Less forgetting loss (Cosine Embedding Loss)
            with torch.no_grad():
                old_features = self.old_self.forward_layerwise(x)["features"]
            loss_less_forgetting = F.cosine_embedding_loss(
                features, old_features, torch.ones(x.size(0), device=self.device)
            )
            loss = loss + self.task_factor * self.lambda_lf * loss_less_forgetting

            # 2. Inter-class separation loss (Margin Ranking Loss)
            new_scores, old_scores = outputs["new_scores"], outputs["old_scores"]
            old_mask = y < self.num_old_classes
            if old_mask.any():
                sel_old_scores = old_scores[old_mask]
                sel_new_scores = new_scores[old_mask]
                sel_y = y[old_mask]

                # Anchor positive: ground truth old class scores
                anchor_pos = sel_old_scores.gather(1, sel_y.view(-1, 1)).expand(
                    -1, self.K
                )
                # Anchor negative: top-K hard negatives from new classes
                anchor_neg, _ = sel_new_scores.topk(self.K, dim=1)

                loss_interclass_separation = F.margin_ranking_loss(
                    anchor_pos,
                    anchor_neg,
                    torch.ones_like(anchor_pos),
                    margin=self.margin,
                )
                loss = loss + loss_interclass_separation

        self.log_dict(
            {
                "train/loss": loss,
                "train/loss_ce": loss_ce,
                "train/loss_forget": loss_less_forgetting or 0.0,
                "train/loss_separation": loss_interclass_separation or 0.0,
                "train/task_factor": self.task_factor,
                "train/classifier_sigma": self.classifier.sigma or 0.0,
                "train/x_mean": x.detach().float().mean(),
                "train/x_var": x.detach().float().var(unbiased=False),
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def on_train_end(self):
        """Update exemplar memory when finishing task training."""
        self.update_memory(self.trainer.datamodule)  # ty: ignore[unresolved-attribute]
