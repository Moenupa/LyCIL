import math

import torch
import torch.nn.functional as F
import lightning as L
from .icarl import ICaRL

from ..data.hfmodule import HFDataModule

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
    # similarities = scale * (similarities - margin)
    similarities = scale * (similarities - margins)

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




def masked_pod_spatial_loss(
    old_fmaps: dict[str, torch.Tensor],
    new_fmaps: dict[str, torch.Tensor],
    normalize: bool = True,
    distill_on_layers: list[str] = ["l1", "l2", "l3", "l4"],
    sample_mask: torch.Tensor = None,
) -> torch.Tensor:
    loss: torch.Tensor = None  # ty: ignore[invalid-assignment]
    sample_mask = sample_mask.to(dtype=torch.bool).view(-1)

    for layer in distill_on_layers:
        a = old_fmaps[layer][sample_mask]
        b = new_fmaps[layer][sample_mask]
        assert a.shape == b.shape, "Shape error"

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        a_h = a.sum(dim=3).view(a.shape[0], -1)
        b_h = b.sum(dim=3).view(b.shape[0], -1)
        a_w = a.sum(dim=2).view(a.shape[0], -1)
        b_w = b.sum(dim=2).view(b.shape[0], -1)

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

def masked_cosine_embedding_loss(
    new_features: torch.Tensor,
    old_features: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    sample_mask = sample_mask.to(device=new_features.device, dtype=torch.bool).view(-1)

    new_features = new_features[sample_mask]
    old_features = old_features[sample_mask]

    target = torch.ones(new_features.shape[0], device=new_features.device)
    return F.cosine_embedding_loss(
        new_features,
        old_features.detach(),
        target,
        reduction="mean",
    )

class PODNet(ICaRL):
    r"""`PODNet`_: Pooled Outputs Distillation for Small-Tasks Incremental Learning. (Douillard et al., ECCV 2020).
    - Exemplar memory: herding + NME-based evaluation
    - Loss :math:`L = L_\text{NCA} + \lambda * \alpha_\text{task} * (L_\text{flat} + L_\text{spatial})`.

    Args:
        lambda_spatial (float, optional): Weight for spatial distillation loss. (default: 5.0)
        lambda_flat (float, optional): Weight for flat distillation loss. (default: 1.0)
        args: See :class:`BaseLearner` for other args.
        kwargs: See :class:`BaseLearner` for other args.

    .. _PODNet:
        https://arxiv.org/abs/2004.13513
    """

    def __init__(
        self,
        *args,
        lambda_spatial: float = 5.0,
        lambda_flat: float = 1.0,
        using_distill: bool = True,
        buffer_training: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lambda_spatial = float(lambda_spatial)
        self.lambda_flat = float(lambda_flat)
        self.using_distill = using_distill
        self.buffer_training = buffer_training

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]

        # Select stage-specific key for optimizer/scheduler configs.
        # If buffer_training is True, prefer "buffer" configs; otherwise use task_id configs.
        stage_key = "buffer" if self.buffer_training else self.task_id


        # Waterfall lookup: stage_key -> default -> {}
        optim_kwargs = (
                self.per_task_optim_args.get(stage_key)
                or self.per_task_optim_args.get("default")
                or {}
        )
        sched_kwargs = (
                self.per_task_sched_args.get(stage_key)
                or self.per_task_sched_args.get("default")
                or {}
        )

        optim = self._get_optimizer(params, **optim_kwargs)
        # If sched_kwargs is None (or explicitly disabled), return optimizer only
        if not sched_kwargs or sched_kwargs.get("type") in (None, "none", "None"):
            return optim

        sched = self._get_scheduler(optim, **sched_kwargs)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


    @property
    def task_factor(self) -> float:
        if self.task_id == 0:
            return 0

        return math.sqrt(
            self.num_seen_classes / (self.num_seen_classes - self.num_old_classes)
        )




    # def training_step(
    #     self, batch: dict[str, torch.Tensor], batch_idx: int
    # ) -> torch.Tensor:
    #     x, y = self.unpack_batch(batch)
    #
    #     new_fmap = self.forward_layerwise(x)
    #
    #     # ce on all classes
    #     loss_lsc = nca(new_fmap["logits"], y)
    #
    #     if self.using_distill:
    #         # distill on old classes ($trainset \setminus cur$)
    #         with torch.no_grad():
    #             old_fmap = self.old_self.forward_layerwise(x)
    #         loss_flat = F.cosine_embedding_loss(
    #             new_fmap["features"],
    #             old_fmap["features"].detach(),
    #             torch.ones(x.shape[0]).to(self.device),
    #         )
    #         loss_spatial = pod_spatial_loss(old_fmap, new_fmap)
    #
    #         loss = loss_lsc + self.task_factor * (
    #             self.lambda_spatial * loss_spatial + self.lambda_flat * loss_flat
    #         )
    #     else:
    #         loss_spatial = None
    #         loss_flat = None
    #         loss = loss_lsc
    #
    #
    #     self.log_dict(
    #         {
    #             "train/loss": loss,
    #             "train/lsc": loss_lsc,
    #             "train/flat": loss_flat or 0.0,
    #             "train/spatial": loss_spatial or 0.0,
    #             "train/classifier_sigma": self.classifier.sigma or 0.0,
    #             # "train/x_mean": x.detach().float().mean(),
    #             # "train/x_var": x.detach().float().var(unbiased=False),
    #         },
    #         prog_bar=True,
    #         on_epoch=True,
    #         on_step=False,
    #         sync_dist=True,
    #     )
    #     return loss

    def _build_distill_mask(self, y: torch.Tensor) -> torch.Tensor | None:
        # 只对旧类样本蒸馏
        if self.num_old_classes <= 0:
            return None
        return y < self.num_old_classes

    def training_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = self.unpack_batch(batch)

        new_fmap = self.forward_layerwise(x)

        # ce on all classes
        loss_lsc = nca(new_fmap["logits"], y)

        if self.using_distill:
            loss_flat = x.new_zeros(())
            loss_spatial = x.new_zeros(())
            distill_mask = self._build_distill_mask(y)
            if distill_mask is not None and distill_mask.any().item():
                # distill on old classes ($trainset \setminus cur$)
                with torch.no_grad():
                    old_fmap = self.old_self.forward_layerwise(x)

                loss_flat = masked_cosine_embedding_loss(
                    new_fmap["features"],
                    old_fmap["features"],
                    sample_mask=distill_mask,
                )

                loss_spatial = masked_pod_spatial_loss(
                    old_fmap,
                    new_fmap,
                    sample_mask=distill_mask,
                )
            loss = loss_lsc + self.task_factor * (
                    self.lambda_spatial * loss_spatial + self.lambda_flat * loss_flat
            )

        else:
            loss_spatial = None
            loss_flat = None
            loss = loss_lsc

        self.log_dict(
            {
                "train/loss": loss,
                "train/lsc": loss_lsc,
                "train/flat": loss_flat or 0.0,
                "train/spatial": loss_spatial or 0.0,
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

    def setup(self, stage) -> None:
        L.LightningModule.setup(self, stage)
        if stage == "fit":
            if self.buffer_training:
                return
            else:
                dm: HFDataModule = self.trainer.datamodule
                self.sync_with_datamodule(dm)

    def on_train_end(self):
        if self.buffer_training:
            return
        else: # already implemented in ICaRL
            dm = self.trainer.datamodule
            # update memory after training current task data, not after replay memory
            # if dm.train_filter_fn is None:
            self.update_memory(dm)

    def on_fit_end(self):
        self.snapshot_old()
