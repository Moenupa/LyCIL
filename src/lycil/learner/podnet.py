import torch
import torch.nn.functional as F

from ..constants import _X_COLUMN_NAME, _Y_COLUMN_NAME
from ..data.buffer import compute_nme
from ..data.hfmodule import HFDataModule
from .base import BaseLearner
import copy


class PODNet(BaseLearner):
    """`iCaRL`_: Incremental Classifier and Representation Learning. (Rebuffi et al., CVPR 2017).
    - Exemplar memory: herding + NME-based evaluation
    - Loss :math:`L = L_\text{CE} + \lambda * L_\text{distill}`.

    Args:
        distill_T (float, optional): Temperature for distillation. Default: 2.0.
        lambda_distill (float, optional): Weight for distillation loss. Default: 1.0.
        args: See :class:`BaseLearner` for other args.
        kwargs: See :class:`BaseLearner` for other args.

    .. _iCaRL:
        https://arxiv.org/abs/1611.07725
    """

    def __init__(
        self,
        *args,
        distill_T: float = 2.0,
        distill_lambda: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.distill_T = float(distill_T)
        self.distill_lambda = float(distill_lambda)



    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)["features"]

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x["features"])
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self



    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # labels belong to current classes only
        x = batch[_X_COLUMN_NAME]
        y = batch[_Y_COLUMN_NAME]
        logits: torch.Tensor = self(x)

        # ce on all classes
        loss_ce = F.cross_entropy(logits, y)

        if self.prev_model is not None and self.task_id > 0:
            # distill on old classes ($trainset \setminus cur$)
            prev_logits = self.forward_prev(x)
            T = self.distill_T

            # mask to only allow old classes in
            p = F.log_softmax(logits[:, : self.num_old_classes] / T, dim=1)
            q = F.softmax(prev_logits[:, : self.num_old_classes] / T, dim=1)
            loss_distill = F.kl_div(p, q, reduction="batchmean") * (T * T)

            loss = loss_ce + self.distill_lambda * loss_distill
        else:
            # first task, no distill
            loss_distill = None
            loss = loss_ce

        self.log_dict(
            {
                "train/loss": loss,
                "train/ce": loss_ce,
                "train/distill": loss_distill or 0.0,
            },
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_train_end(self):
        self.update_memory(self.trainer.datamodule)  # ty: ignore[unresolved-attribute]

    def on_fit_end(self):
        self.snapshot_prev()

    @torch.no_grad()
    def update_memory(self, dm: HFDataModule, **kwargs) -> None:
        if dm.buffer is None:
            raise RuntimeError("Buffer is not initialized.")

        self.eval()
        if dm.buffer.is_adaptive:
            # vacate exemplars for more classes
            dm.buffer.reduce_exemplars(dm.buffer.size_per_class(self.num_seen_classes))
            self._construct_exemplar(dm, **kwargs)
        else:
            self._construct_exemplar_unified(dm, **kwargs)
        self.train()
        return

    @torch.no_grad()
    def _construct_exemplar(self, dm: HFDataModule, **kwargs) -> None:
        raise NotImplementedError

        assert dm.buffer is not None
        # construct exemplar set for current classes
        for class_idx in range(self.num_old_classes, self.num_seen_classes):
            pass

    @torch.no_grad()
    def _construct_exemplar_unified(self, dm: HFDataModule, **kwargs) -> None:
        # for dataloader during exemplar construction,
        # rather conservative because args are hard-coded here
        loader_kwargs = dict(
            batch_size=1,
            shuffle=False,
            num_workers=8,
        )

        assert dm.buffer is not None
        per_class_means = {}

        # find means of old classes with newly trained network
        for class_idx in range(self.num_old_classes):
            loader = dm.buffer.get_dataloader(
                keys=[f"{class_idx}"],
                transform_name=dm.get_effective_transform_name(),
                loader_kwargs=loader_kwargs,
            )
            mean, _ = compute_nme(loader, self.feature_extractor, self.device)
            per_class_means[class_idx] = mean

        # construct exemplar set for current classes
        for class_idx in range(self.num_old_classes, self.num_seen_classes):
            # 1. single pass on all data
            train_loader = dm.get_dataloader(
                split=dm._split_train,
                filter_fn=lambda e: e[_Y_COLUMN_NAME] == class_idx,
                transform_name=dm.get_effective_transform_name(),
                loader_kwargs=loader_kwargs,
            )
            mean, per_sample_features = compute_nme(
                train_loader, self.feature_extractor, self.device
            )

            # 2. select exemplars by herding
            # for now, use first m samples
            m = dm.buffer.size_per_class(self.num_seen_classes)
            selected_idx = list(range(0, m))
            # TODO: implement full herding
            # herding implementation from another library is below:
            # selected_exemplars = []
            # exemplar_vectors = []
            # for k in range(1, m + 1):
            #     S = np.sum(
            #         exemplar_vectors, axis=0
            #     )  # [feature_dim] sum of selected exemplars vectors
            #     mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
            #     i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

            #     selected_exemplars.append(
            #         np.array(data[i])
            #     )  # New object to avoid passing by inference
            #     exemplar_vectors.append(
            #         np.array(vectors[i])
            #     )  # New object to avoid passing by inference

            #     vectors = np.delete(
            #         vectors, i, axis=0
            #     )  # Remove it to avoid duplicative selection
            #     data = np.delete(
            #         data, i, axis=0
            #     )  # Remove it to avoid duplicative selection
            selected_dataset = dm.get_filtered_dataset(
                split=dm._split_train,
                filter_fn=lambda e: e[_Y_COLUMN_NAME] == class_idx,
            ).select(selected_idx)
            dm.buffer[f"{class_idx}"] = selected_dataset

            # 3. recompute class mean after selection
            loader = dm.buffer.get_dataloader(
                keys=[f"{class_idx}"],
                transform_name=dm.get_effective_transform_name(),
                loader_kwargs=loader_kwargs,
            )
            mean, _ = compute_nme(loader, self.feature_extractor, self.device)
            per_class_means[class_idx] = mean

        dm.buffer.per_class_means = per_class_means

        return


    """
        def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)  # [bs, 64, 32, 32]

        x_1 = self.layer1(x)  # [bs, 128, 32, 32]
        x_2 = self.layer2(x_1)  # [bs, 256, 16, 16]
        x_3 = self.layer3(x_2)  # [bs, 512, 8, 8]
        x_4 = self.layer4(x_3)  # [bs, 512, 4, 4]

        pooled = self.avgpool(x_4)  # [bs, 512, 1, 1]
        features = torch.flatten(pooled, 1)  # [bs, 512]
        # x = self.fc(x)

        return {
            'fmaps': [x_1, x_2, x_3, x_4],
            'features': features
        }

    def forward(self, x):
        return self._forward_impl(x)

    @property
    def last_conv(self):
        if hasattr(self.layer4[-1], 'conv3'):
            return self.layer4[-1].conv3
        else:
            return self.layer4[-1].conv2
    """
