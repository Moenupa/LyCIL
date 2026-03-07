import torch
import torch.nn.functional as F

from ..constants import _Y_COLUMN_NAME
from ..data.buffer import compute_nme
from ..data.hfmodule import HFDataModule
from .base import BaseLearner


class ICaRL(BaseLearner):
    r"""`iCaRL`_: Incremental Classifier and Representation Learning. (Rebuffi et al., CVPR 2017).
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

    def training_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = self.unpack_batch(batch)
        logits: torch.Tensor = self(x)

        # ce on all classes
        loss_ce = F.cross_entropy(logits, y)

        if self.task_id > 0:
            # distill on old classes ($trainset \setminus cur$)
            old_logits = self.old_self.forward_no_grad(x)
            T = self.distill_T

            # mask to only allow old classes in
            p = F.log_softmax(logits[:, : self.num_old_classes] / T, dim=1)
            q = F.softmax(old_logits[:, : self.num_old_classes] / T, dim=1)
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

    # @torch.no_grad()
    # def _construct_exemplar_unified(self, dm: HFDataModule, **kwargs) -> None:
    #     # for dataloader during exemplar construction,
    #     # rather conservative because args are hard-coded here
    #     loader_kwargs = dict(
    #         batch_size=1,
    #         shuffle=False,
    #         num_workers=8,
    #     )
    #
    #     assert dm.buffer is not None
    #     per_class_means = {}
    #
    #     # find means of old classes with newly trained network
    #     # for class_idx in range(self.num_old_classes):
    #     #     loader = dm.buffer.get_dataloader(
    #     #         keys=[f"{class_idx}"],
    #     #         transform_name=dm.get_effective_transform_name(),
    #     #         loader_kwargs=loader_kwargs,
    #     #     )
    #     #     mean, _ = compute_nme(loader, self.feature_extractor, self.device)
    #     #     per_class_means[class_idx] = mean
    #
    #     # construct exemplar set for current classes
    #     for class_idx in range(self.num_old_classes, self.num_seen_classes):
    #         # import pdb;pdb.set_trace()
    #         # 1. single pass on all data
    #         # train_loader = dm.get_dataloader(
    #         #     split=dm._split_train,
    #         #     filter_fn=lambda e: e[_Y_COLUMN_NAME] == class_idx,
    #         #     transform_name=dm.get_effective_transform_name(),
    #         #     loader_kwargs=loader_kwargs,
    #         # )
    #         # mean, per_sample_features = compute_nme(
    #         #     train_loader, self.feature_extractor, self.device
    #         # )
    #
    #         # 2. select exemplars by herding
    #         # for now, use first m samples
    #         m = dm.buffer.size_per_class(self.num_seen_classes)
    #         selected_idx = list(range(0, m))
    #         # TODO: implement full herding
    #         # herding implementation from another library is below:
    #         # selected_exemplars = []
    #         # exemplar_vectors = []
    #         # for k in range(1, m + 1):
    #         #     S = np.sum(
    #         #         exemplar_vectors, axis=0
    #         #     )  # [feature_dim] sum of selected exemplars vectors
    #         #     mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
    #         #     i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
    #
    #         #     selected_exemplars.append(
    #         #         np.array(data[i])
    #         #     )  # New object to avoid passing by inference
    #         #     exemplar_vectors.append(
    #         #         np.array(vectors[i])
    #         #     )  # New object to avoid passing by inference
    #
    #         #     vectors = np.delete(
    #         #         vectors, i, axis=0
    #         #     )  # Remove it to avoid duplicative selection
    #         #     data = np.delete(
    #         #         data, i, axis=0
    #         #     )  # Remove it to avoid duplicative selection
    #         selected_dataset = dm.get_filtered_dataset(
    #             split=dm._split_train,
    #             filter_fn=lambda e: e[_Y_COLUMN_NAME] == class_idx,
    #         ).select(selected_idx)
    #         selected_dataset.reset_format()
    #         dm.buffer[f"{class_idx}"] = selected_dataset
    #
    #         # 3. recompute class mean after selection
    #         # TODO: fix bug of  Data Transform
    #         # loader = dm.buffer.get_dataloader(
    #         #     keys=[f"{class_idx}"],
    #         #     transform_name=dm.get_effective_transform_name(),
    #         #     loader_kwargs=loader_kwargs,
    #         # )
    #         # mean, _ = compute_nme(loader, self.feature_extractor, self.device)
    #         # per_class_means[class_idx] = mean
    #
    #     dm.buffer.per_class_means = per_class_means
    #
    #     return

    @torch.no_grad()
    def _construct_exemplar_unified(self, dm: HFDataModule, **kwargs) -> None:
        """
        Unified exemplar construction for fixed-per-class buffer.

        支持两种策略：
        - exemplar_selection="herding"  (默认)
        - exemplar_selection="random"

        可选参数：
        - exemplar_seed: int = 42
        - exemplar_transform_mode: str = "test"
          herding / mean 计算时建议固定用 test transform，避免随机增强干扰
        """
        loader_kwargs = dict(
            batch_size=128,
            shuffle=False,
            num_workers=8,
        )

        assert dm.buffer is not None

        exemplar_selection = kwargs.get(
            "exemplar_selection",
            getattr(self, "exemplar_selection", "herding"),
        )
        exemplar_seed = int(
            kwargs.get(
                "exemplar_seed",
                getattr(self, "exemplar_seed", 42),
            )
        )

        feature_tfm = dm.get_effective_transform(mode="train")

        if exemplar_selection not in {"random", "herding"}:
            raise ValueError(
                f"Unsupported exemplar_selection={exemplar_selection}, "
                "expected one of {'random', 'herding'}."
            )

        per_class_means = {}

        # 1) 先用当前网络重算旧类的 class mean
        for class_idx in range(self.num_old_classes):
            if f"{class_idx}" not in dm.buffer:
                continue
            if len(dm.buffer[f"{class_idx}"]) == 0:
                continue

            loader = dm.buffer.get_dataloader(
                keys=[f"{class_idx}"],
                transform=feature_tfm,
                loader_kwargs=loader_kwargs,
            )
            mean, _ = compute_nme(loader, self.feature_extractor, self.device)
            mean = F.normalize(mean.unsqueeze(0), dim=1).squeeze(0)
            per_class_means[class_idx] = mean.cpu()

        # 2) 为当前新类构建 exemplar
        for class_idx in range(self.num_old_classes, self.num_seen_classes):
            class_dataset = dm.get_filtered_dataset(
                split=dm._split_train,
                filter_fn=lambda e, c=class_idx: e[_Y_COLUMN_NAME] == c,
            )
            class_dataset.reset_format()

            n_samples = len(class_dataset)
            if n_samples == 0:
                continue

            m = min(dm.buffer.size_per_class(self.num_seen_classes), n_samples)

            if exemplar_selection == "random":
                # 真随机：不是取前 m 个
                g = torch.Generator()
                g.manual_seed(exemplar_seed + int(class_idx))
                selected_idx = torch.randperm(n_samples, generator=g)[:m].tolist()

            else:
                # 真 herding
                train_loader = dm.get_dataloader(
                    split=dm._split_train,
                    filter_fn=lambda e, c=class_idx: e[_Y_COLUMN_NAME] == c,
                    transform=feature_tfm,
                    loader_kwargs=loader_kwargs,
                    use_buffer=False,
                )
                class_mean, per_sample_features = compute_nme(
                    train_loader, self.feature_extractor, self.device
                )

                # 归一化 class mean，和 per-sample normalized features 对齐
                class_mean = F.normalize(class_mean.unsqueeze(0), dim=1).squeeze(0).cpu()
                feats = per_sample_features.cpu()  # [n, d]

                selected_idx = []
                selected_mask = torch.zeros(n_samples, dtype=torch.bool)
                running_sum = torch.zeros_like(class_mean)

                for k in range(1, m + 1):
                    candidate_idx = (~selected_mask).nonzero(as_tuple=False).squeeze(1)
                    candidate_feats = feats[candidate_idx]  # [n_remain, d]

                    # iCaRL herding:
                    # 选使 ||mu - (S + x)/k|| 最小的样本
                    mu_p = (running_sum.unsqueeze(0) + candidate_feats) / k
                    dist = torch.norm(class_mean.unsqueeze(0) - mu_p, p=2, dim=1)
                    best_rel = torch.argmin(dist).item()
                    best_abs = candidate_idx[best_rel].item()

                    selected_idx.append(best_abs)
                    selected_mask[best_abs] = True
                    running_sum += feats[best_abs]

            selected_dataset = class_dataset.select(selected_idx)
            selected_dataset.reset_format()
            dm.buffer[f"{class_idx}"] = selected_dataset

            # 3) 用选后的 exemplar 重算该类 mean
            loader = dm.buffer.get_dataloader(
                keys=[f"{class_idx}"],
                transform=feature_tfm,
                loader_kwargs=loader_kwargs,
            )
            mean, _ = compute_nme(loader, self.feature_extractor, self.device)
            mean = F.normalize(mean.unsqueeze(0), dim=1).squeeze(0)
            per_class_means[class_idx] = mean.cpu()

        dm.buffer.per_class_means = per_class_means
        return
