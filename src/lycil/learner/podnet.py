import math

import torch
import torch.nn.functional as F

from .icarl import ICaRL


def nca(
    similarities: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    scale: float = 1.0,
    margin: float = 0.6,
    exclude_pos_denominator: bool = True,
    hinge_proxynca: bool = False,
) -> torch.Tensor:
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if exclude_pos_denominator:
        similarities = similarities - similarities.max(1)[0].view(-1, 1)

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)), targets] = similarities[
            torch.arange(len(similarities)), targets
        ]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.0)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(
        similarities, targets, weight=class_weights, reduction="mean"
    )


def pod_spatial_loss(
    old_fmaps: dict[str, torch.Tensor],
    new_fmaps: dict[str, torch.Tensor],
    normalize: bool = True,
    distill_on_layers: list[str] = ["l1", "l2", "l3", "l4"],
) -> torch.Tensor:
    loss: torch.Tensor = None  # ty: ignore[invalid-assignment]
    for layer in distill_on_layers:
        a = old_fmaps[layer]
        b = new_fmaps[layer]
        assert a.shape == b.shape, "Shape error"

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        a_h = a.sum(dim=3).view(a.shape[0], -1)  # [bs, c*w]
        b_h = b.sum(dim=3).view(b.shape[0], -1)  # [bs, c*w]
        a_w = a.sum(dim=2).view(a.shape[0], -1)  # [bs, c*h]
        b_w = b.sum(dim=2).view(b.shape[0], -1)  # [bs, c*h]

        a = torch.cat([a_h, a_w], dim=-1)
        b = torch.cat([b_h, b_w], dim=-1)

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        if loss is None:
            loss = layer_loss
        else:
            loss += layer_loss

    return loss / len(distill_on_layers)


class PODNet(ICaRL):
    r"""`PODNet`_: Pooled Outputs Distillation for Small-Tasks Incremental Learning. (Douillard et al., ECCV 2020).
    - Exemplar memory: herding + NME-based evaluation
    - Loss :math:`L = L_\text{NCA} + \lambda * \alpha_\text{task} * (L_\text{flat} + L_\text{spatial})`.

    Args:
        lambda_distill (float, optional): Weight for distillation loss. Default: 1.0.
        args: See :class:`BaseLearner` for other args.
        kwargs: See :class:`BaseLearner` for other args.

    .. _PODNet:
        https://arxiv.org/abs/2004.13513
    """

    def __init__(
        self,
        *args,
        distill_lambda: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.distill_lambda = float(distill_lambda)

    @property
    def task_factor(self) -> float:
        if self.task_id == 0:
            return 0

        return math.sqrt(
            self.num_seen_classes / (self.num_seen_classes - self.num_old_classes)
        )

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = self.unpack_batch(batch)
        new_fmap = self.forward_layerwise(x)

        # ce on all classes
        loss_lsc = nca(new_fmap["logits"], y)

        if self.task_id > 0:
            # distill on old classes ($trainset \setminus cur$)
            with torch.no_grad():
                old_fmap = self.old_self.forward_layerwise(x)
            loss_flat = F.cosine_embedding_loss(
                new_fmap["features"],
                old_fmap["features"].detach(),
                torch.ones(x.shape[0]).to(self._device),
            )
            loss_spatial = pod_spatial_loss(new_fmap, old_fmap)

            loss = loss_lsc + self.task_factor * self.distill_lambda * (
                loss_spatial + loss_flat
            )
        else:
            # first task, no distill
            loss_spatial = None
            loss_flat = None
            loss = loss_lsc

        self.log_dict(
            {
                "train/loss": loss,
                "train/lsc": loss_lsc,
                "train/flat": loss_spatial or 0.0,
                "train/spatial": loss_flat or 0.0,
            },
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_train_end(self):
        # already implemented in ICaRL
        dm = self.trainer.datamodule  # ty: ignore[unresolved-attribute]

        # update memory after training current task data, not after replay memory
        if dm.train_filter_fn is None:
            self.update_memory(dm)

    def on_fit_end(self):
        self.snapshot_old()
