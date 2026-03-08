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

        per_class_quota = dm.buffer.size_per_class(self.num_seen_classes)

        self.eval()
        if dm.buffer.is_adaptive:
            # Adaptive memory: reuse the full global budget on the currently
            # seen classes, then shrink old classes as new ones arrive.
            dm.buffer.reduce_exemplars(per_class_quota)
        self._construct_exemplar_unified(dm, per_class_quota=per_class_quota, **kwargs, )
        self.train()
        return


    # @torch.no_grad()
    # def _construct_exemplar_unified(self, dm: HFDataModule, **kwargs) -> None:
    #     """
    #     统一的 exemplar 构建实现。
    #
    #     同时支持两种 buffer 策略：
    #     - 总内存自适应模式（`mem_size`）
    #     - 固定每类样本数模式（`mem_size_per_class`）
    #
    #     调用约定：
    #     - 自适应总内存模式下，应当在调用本函数前先把旧类 exemplar 裁剪到新的配额；
    #     - 固定每类样本数模式下，可以直接调用本函数，对旧类重算 mean，并为新类构建 exemplar。
    #     """
    #     # exemplar 构建时使用的 dataloader 配置
    #     # 这里固定不用 shuffle，避免 exemplar 选择过程不稳定
    #     loader_kwargs = dict(
    #         batch_size=128,
    #         shuffle=False,
    #         num_workers=8,
    #     )
    #
    #     assert dm.buffer is not None
    #
    #     # exemplar 选择策略，默认 herding
    #     exemplar_selection = kwargs.get(
    #         "exemplar_selection",
    #         getattr(self, "exemplar_selection", "herding"),
    #     )
    #     # 随机选择时使用的随机种子
    #     exemplar_seed = int(
    #         kwargs.get(
    #             "exemplar_seed",
    #             getattr(self, "exemplar_seed", 42),
    #         )
    #     )
    #     # 当前每个类别允许保留的 exemplar 数量
    #     # 对于 mem_size 模式，这里会根据当前已见类别数动态计算
    #     per_class_quota = int(
    #         kwargs.get(
    #             "per_class_quota",
    #             dm.buffer.size_per_class(self.num_seen_classes),
    #         )
    #     )
    #
    #     # 统一使用 test transform 提特征，避免随机增强影响 exemplar 选择和类均值计算
    #     feature_tfm = dm.get_effective_transform(mode="test")
    #
    #     if exemplar_selection not in {"random", "herding"}:
    #         raise ValueError(
    #             f"Unsupported exemplar_selection={exemplar_selection}, "
    #             "expected one of {'random', 'herding'}."
    #         )
    #
    #     per_class_means = {}
    #
    #     # 1) 用最新网络重新计算旧类别的类中心
    #     for class_idx in range(self.num_old_classes):
    #         if f"{class_idx}" not in dm.buffer:
    #             continue
    #         if len(dm.buffer[f"{class_idx}"]) == 0:
    #             continue
    #
    #         loader = dm.buffer.get_dataloader(
    #             keys=[f"{class_idx}"],
    #             transform=feature_tfm,
    #             loader_kwargs=loader_kwargs,
    #         )
    #         mean, _ = compute_nme(loader, self.feature_extractor, self.device)
    #         mean = F.normalize(mean.unsqueeze(0), dim=1).squeeze(0)
    #         per_class_means[class_idx] = mean.cpu()
    #
    #     # 2) 为当前新类别构建 exemplar 集合
    #     for class_idx in range(self.num_old_classes, self.num_seen_classes):
    #         class_dataset = dm.get_filtered_dataset(
    #             split=dm._split_train,
    #             filter_fn=lambda e, c=class_idx: e[_Y_COLUMN_NAME] == c,
    #         )
    #         class_dataset.reset_format()
    #
    #         n_samples = len(class_dataset)
    #         if n_samples == 0:
    #             continue
    #
    #         # 当前类实际保留的样本数，不能超过该类真实样本数
    #         m = min(per_class_quota, n_samples)
    #
    #         # 2.1 随机选择 exemplar
    #         if exemplar_selection == "random":
    #             g = torch.Generator()
    #             g.manual_seed(exemplar_seed + int(class_idx))
    #             selected_idx = torch.randperm(n_samples, generator=g)[:m].tolist()
    #
    #         # 2.2 herding 选择 exemplar
    #         else:
    #             # 先对当前类别全量样本做一次前向，提取特征
    #             train_loader = dm.get_dataloader(
    #                 split=dm._split_train,
    #                 filter_fn=lambda e, c=class_idx: e[_Y_COLUMN_NAME] == c,
    #                 transform=feature_tfm,
    #                 loader_kwargs=loader_kwargs,
    #                 use_buffer=False,
    #             )
    #             class_mean, per_sample_features = compute_nme(
    #                 train_loader, self.feature_extractor, self.device
    #             )
    #
    #             class_mean = F.normalize(class_mean.unsqueeze(0), dim=1).squeeze(0).cpu()
    #             feats = per_sample_features.cpu()
    #
    #             selected_idx = []
    #             selected_mask = torch.zeros(n_samples, dtype=torch.bool)
    #             running_sum = torch.zeros_like(class_mean)
    #
    #             # 按 herding 规则逐个选择，使 exemplar 均值尽量逼近类中心
    #             for k in range(1, m + 1):
    #                 candidate_idx = (~selected_mask).nonzero(as_tuple=False).squeeze(1)
    #                 candidate_feats = feats[candidate_idx]
    #
    #                 mu_p = (running_sum.unsqueeze(0) + candidate_feats) / k
    #                 dist = torch.norm(class_mean.unsqueeze(0) - mu_p, p=2, dim=1)
    #                 best_rel = torch.argmin(dist).item()
    #                 best_abs = candidate_idx[best_rel].item()
    #
    #                 selected_idx.append(best_abs)
    #                 selected_mask[best_abs] = True
    #                 running_sum += feats[best_abs]
    #
    #         # 3) 将选中的 exemplar 写入 buffer
    #         selected_dataset = class_dataset.select(selected_idx)
    #         selected_dataset.reset_format()
    #         dm.buffer[f"{class_idx}"] = selected_dataset
    #
    #         # 4) 用选中的 exemplar 重新计算该类的类中心
    #         loader = dm.buffer.get_dataloader(
    #             keys=[f"{class_idx}"],
    #             transform=feature_tfm,
    #             loader_kwargs=loader_kwargs,
    #         )
    #         mean, _ = compute_nme(loader, self.feature_extractor, self.device)
    #         mean = F.normalize(mean.unsqueeze(0), dim=1).squeeze(0)
    #         per_class_means[class_idx] = mean.cpu()
    #
    #     # 保存所有类别的类中心，供 NME 推理使用
    #     dm.buffer.per_class_means = per_class_means
    #     return

    @torch.no_grad()
    def _construct_exemplar_unified(self, dm: HFDataModule, **kwargs) -> None:
        """
        统一的 exemplar 构建实现。

        同时支持两种 buffer 策略：
        - 总内存自适应模式（`mem_size`）
        - 固定每类样本数模式（`mem_size_per_class`）

        调用约定：
        - 自适应总内存模式下，应当在调用本函数前先把旧类 exemplar 裁剪到新的配额；
        - 固定每类样本数模式下，可以直接调用本函数，对旧类重算 mean，并为新类构建 exemplar。
        """
        # exemplar 构建时使用的 dataloader 配置
        # 这里固定不用 shuffle，避免 exemplar 选择过程不稳定
        loader_kwargs = dict(
            batch_size=128,
            shuffle=False,
            num_workers=8,
        )

        assert dm.buffer is not None

        # exemplar 选择策略，默认 herding
        exemplar_selection = kwargs.get(
            "exemplar_selection",
            getattr(self, "exemplar_selection", "herding"),
        )
        # 随机选择时使用的随机种子
        exemplar_seed = int(
            kwargs.get(
                "exemplar_seed",
                getattr(self, "exemplar_seed", 42),
            )
        )
        # 当前每个类别允许保留的 exemplar 数量
        # 对于 mem_size 模式，这里会根据当前已见类别数动态计算
        per_class_quota = int(
            kwargs.get(
                "per_class_quota",
                dm.buffer.size_per_class(self.num_seen_classes),
            )
        )

        # 统一使用 test transform 提特征，避免随机增强影响 exemplar 选择和类均值计算
        feature_tfm = dm.get_effective_transform(mode="test")

        if exemplar_selection not in {"random", "herding"}:
            raise ValueError(
                f"Unsupported exemplar_selection={exemplar_selection}, "
                "expected one of {'random', 'herding'}."
            )

        per_class_means = {}

        # 1) 用最新网络重新计算旧类别的类中心
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

        # 2) 先一次性筛出当前 task 的训练子集，避免每个类别都对整份训练集重复 filter
        task_train_dataset = dm.get_filtered_dataset(
            split=dm._split_train,
            filter_fn=lambda e: self.num_old_classes <= e[_Y_COLUMN_NAME] < self.num_seen_classes,
            transform=feature_tfm,
            use_buffer=False,
        )

        # 3) 在当前 task 子集内，先建立 “类别 -> 样本索引列表” 的映射
        #    这样后面每个类别直接 select 对应索引即可，不再重复 filter
        class_to_indices = {class_idx: [] for class_idx in range(self.num_old_classes, self.num_seen_classes)}
        task_labels = task_train_dataset[_Y_COLUMN_NAME]
        for sample_idx, y in enumerate(task_labels):
            class_to_indices[int(y)].append(sample_idx)

        # 4) 为当前新类别构建 exemplar 集合
        for class_idx in range(self.num_old_classes, self.num_seen_classes):
            class_indices = class_to_indices.get(class_idx, [])
            if len(class_indices) == 0:
                continue

            # 从当前 task 子集中切出该类别的数据
            class_dataset = task_train_dataset.select(class_indices)

            n_samples = len(class_dataset)
            if n_samples == 0:
                continue

            # 当前类实际保留的样本数，不能超过该类真实样本数
            m = min(per_class_quota, n_samples)

            # 4.1 随机选择 exemplar
            if exemplar_selection == "random":
                g = torch.Generator()
                g.manual_seed(exemplar_seed + int(class_idx))
                selected_idx = torch.randperm(n_samples, generator=g)[:m].tolist()

            # 4.2 herding 选择 exemplar
            else:
                # 直接在当前类别子集上做一次前向，提取特征
                train_loader = torch.utils.data.DataLoader(class_dataset, **loader_kwargs)
                class_mean, per_sample_features = compute_nme(
                    train_loader, self.feature_extractor, self.device
                )

                class_mean = F.normalize(class_mean.unsqueeze(0), dim=1).squeeze(0).cpu()
                feats = per_sample_features.cpu()

                selected_idx = []
                selected_mask = torch.zeros(n_samples, dtype=torch.bool)
                running_sum = torch.zeros_like(class_mean)

                # 按 herding 规则逐个选择，使 exemplar 均值尽量逼近类中心
                for k in range(1, m + 1):
                    candidate_idx = (~selected_mask).nonzero(as_tuple=False).squeeze(1)
                    candidate_feats = feats[candidate_idx]

                    mu_p = (running_sum.unsqueeze(0) + candidate_feats) / k
                    dist = torch.norm(class_mean.unsqueeze(0) - mu_p, p=2, dim=1)
                    best_rel = torch.argmin(dist).item()
                    best_abs = candidate_idx[best_rel].item()

                    selected_idx.append(best_abs)
                    selected_mask[best_abs] = True
                    running_sum += feats[best_abs]

            # 5) 将选中的 exemplar 写入 buffer
            # 注意：这里 class_dataset 目前带有 test transform，写入 buffer 前要 reset_format 恢复原始数据
            selected_dataset = class_dataset.select(selected_idx)
            selected_dataset.reset_format()
            dm.buffer[f"{class_idx}"] = selected_dataset

            # 6) 用选中的 exemplar 重新计算该类的类中心
            loader = dm.buffer.get_dataloader(
                keys=[f"{class_idx}"],
                transform=feature_tfm,
                loader_kwargs=loader_kwargs,
            )
            mean, _ = compute_nme(loader, self.feature_extractor, self.device)
            mean = F.normalize(mean.unsqueeze(0), dim=1).squeeze(0)
            per_class_means[class_idx] = mean.cpu()

        # 保存所有类别的类中心，供 NME 推理使用
        dm.buffer.per_class_means = per_class_means
        return