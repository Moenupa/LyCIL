import torch
import torch.nn.functional as F
from torch.linalg import vector_norm

from ..backbone import DERNetBackbone
from ..classifier.linears import SimpleLinear
from .base import BaseLearner


class DER(BaseLearner):
    """`DER`_: Dynamically Expandable Representation. (Yan et al., CVPR 2021).

    - Dynamic expansion of feature extractors for new tasks.
    - Auxiliary classifier for the current task classes.

    .. _DER:
        https://arxiv.org/abs/2103.16788
    """

    def __init__(
        self,
        **kwargs,
    ):
        # must use DERNetBackbone
        kwargs["backbone_cls"] = DERNetBackbone
        # must use linear head see self.weight_align()
        kwargs["head"] = "linear"
        super().__init__(**kwargs)

        if self.head_type != "linear":
            raise NotImplementedError("DER currently only supports linear head.")

        # type annotations to boost development
        self.backbone: DERNetBackbone
        self.classifier: SimpleLinear
        self.aux_classifier: SimpleLinear

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = self.unpack_batch(batch)
        outputs = self.forward_layerwise(x)

        logits = outputs["logits"]
        loss_ce = F.cross_entropy(logits, y)

        if self.task_id > 0:
            # Auxiliary loss on new classes only
            # Targets: 0 for all old classes, 1..N for new task classes
            aux_targets = torch.where(
                y - self.num_old_classes + 1 > 0,
                y - self.num_old_classes + 1,
                0,
            )
            aux_logits = outputs["aux_logits"]
            loss_aux = F.cross_entropy(aux_logits, aux_targets)
            loss = loss_ce + loss_aux
        else:
            loss_aux = None
            loss = loss_ce

        self.log_dict(
            {
                "train/loss": loss,
                "train/loss_ce": loss_ce,
                "train/loss_aux": loss_aux or 0.0,
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
        # Weight alignment for biased classifier
        if self.task_id > 0:
            self.weight_align()

        self.update_memory(self.trainer.datamodule)  # ty: ignore[unresolved-attribute]

    def forward_layerwise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        fmap = super().forward_layerwise(x)
        aux_logits = self.aux_classifier(fmap["features"][:, -self.backbone.out_dim :])[
            "logits"
        ]

        return fmap | {"aux_logits": aux_logits}

    @torch.no_grad()
    def expand_head(self, out_delta: int, in_delta: int = 0) -> None:
        # dernet backbone expansion
        self.backbone.prepare_for_new_task()
        # aux_classifer should also be re-init, +1 for old class (0) in aux classifier
        self.aux_classifier = SimpleLinear(self.backbone.out_dim, out_delta + 1)
        # normal classifier expansion, with input also expanded for dernet backbone
        super().expand_head(out_delta, self.backbone.out_dim)

    @torch.no_grad()
    def weight_align(self):
        """Align new classes to mitigate bias towards old classes.

        Calculates norms of old & new class weights
        and scales new class weights by old/new ratio.

        Raises:
            NotImplementedError: If classifier is not SimpleLinear.
        """
        if not isinstance(self.classifier, SimpleLinear):
            raise NotImplementedError(
                "Weight alignment only implemented for SimpleLinear head."
            )

        # align weights (new classes) to mitigate bias towards old classes.
        w = self.classifier.weight.data
        # 1d norm for each class (row) in the weight matrix
        mean_new = torch.mean(vector_norm(w[self.num_old_classes :, :], ord=2, dim=1))
        mean_old = torch.mean(vector_norm(w[: self.num_old_classes, :], ord=2, dim=1))
        gamma = mean_old / (mean_new + 1e-8)
        self.classifier.weight.data[self.num_old_classes :, :] *= gamma


    def on_train_epoch_start(self):
        if len(self.backbone.convnets) == 0:
            return

        # 最新块训练
        self.backbone.convnets[-1].train()
        # 冻结旧块参数 + 切 eval
        for conv in self.backbone.convnets[:-1]:
            conv.requires_grad_(False)
            conv.eval()