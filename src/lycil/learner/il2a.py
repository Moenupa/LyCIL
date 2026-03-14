import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from tqdm import tqdm
from .base import BaseLearner
from ..data.hfmodule import HFDataModule


class IL2A(BaseLearner):
    def __init__(
        self,
        *args,
        temp: float = 1.0,
        lambda_fkd: float = 1.0,
        lambda_proto: float = 1.0,
        ratio: float = 1.0,
        alpha: float = 20.0,
        mix_time: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.temp = float(temp)
        self.lambda_fkd = float(lambda_fkd)
        self.lambda_proto = float(lambda_proto)
        self.ratio = float(ratio)
        self.alpha = float(alpha)
        self.mix_time = int(mix_time)

        self.task_size = 0
        self.aux_size = 0
        self.num_old_classes = 0

        self._protos: list[torch.Tensor] = []
        self._covs: list[torch.Tensor] = []

    def sync_with_datamodule(self, dm: "HFDataModule"):
        task_id = dm.get_current_task()
        if self.task_id is not None and task_id == self.task_id:
            return

        self.task_id = task_id

        task_size = dm.num_seen_classes - (self.num_seen_classes or 0)
        if task_size <= 0:
            raise RuntimeError(
                f"Expect positive class expansion, but got {task_size}. "
                f"DataModule has {dm.num_seen_classes} seen classes, while model has "
                f"{self.num_seen_classes} seen classes."
            )

        aux_size = task_size * (task_size - 1) // 2
        self.expand_head(task_size + aux_size)

        self.task_size = task_size
        self.aux_size = aux_size
        self.num_old_classes = self.num_seen_classes or 0
        self.num_seen_classes = dm.num_seen_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_layerwise(x)["logits"][:, : self.num_seen_classes]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = self.unpack_batch(batch)
        x, y = self.class_aug(x, y)

        outs = self.forward_layerwise(x)
        logits = outs["logits"]
        features = outs["features"]

        loss_ce = F.cross_entropy(logits / self.temp, y)
        loss = loss_ce
        loss_distill = None
        loss_proto = None

        if self.task_id > 0:
            with torch.no_grad():
                old_features = self.old_self.forward_layerwise(x)["features"]

            loss_distill = torch.dist(features, old_features, p=2)
            loss_proto = self.prototype_loss(x.shape[0])
            loss = loss + self.lambda_fkd * loss_distill + self.lambda_proto * loss_proto

        self.log_dict(
            {
                "train/loss": loss,
                "train/ce": loss_ce,
                "train/loss_distill": loss_distill or 0.0,
                "train/loss_proto": loss_proto or 0.0,
                "train/x_mean": x.detach().float().mean(),
                "train/x_var": x.detach().float().var(unbiased=False),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_train_end(self):
        self.build_protos(self.trainer.datamodule)

    def class_aug(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.task_size <= 1:
            return x, y

        mask = y >= self.num_old_classes
        x_new = x[mask]
        y_new = y[mask]
        if len(y_new) < 2:
            return x, y

        beta = torch.distributions.Beta(self.alpha, self.alpha)
        mix_x, mix_y = [], []

        for _ in range(self.mix_time):
            index = torch.randperm(len(y_new), device=y_new.device)
            x_perm = x_new[index]
            y_perm = y_new[index]

            diff = y_new != y_perm
            if not diff.any():
                continue

            x_a, y_a = x_new[diff], y_new[diff]
            x_b, y_b = x_perm[diff], y_perm[diff]

            lam = beta.sample((len(y_a),)).to(device=x.device, dtype=x.dtype)
            lam = torch.where((lam < 0.4) | (lam > 0.6), 0.5, lam)
            lam = lam[:, None, None, None]

            mix_x.append(lam * x_a + (1.0 - lam) * x_b)
            mix_y.append(self.map_targets(y_a, y_b))

        if not mix_x:
            return x, y

        x = torch.cat([x, *mix_x], dim=0)
        y = torch.cat([y, *mix_y], dim=0)
        return x, y

    def map_targets(self, y_a: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        y_large = torch.maximum(y_a, y_b) - self.num_old_classes
        y_small = torch.minimum(y_a, y_b) - self.num_old_classes
        return (y_large * (y_large - 1) // 2 + y_small + self.num_seen_classes).long()

    def prototype_loss(self, batch_size: int) -> torch.Tensor:
        if self.num_old_classes == 0 or not self._protos:
            return torch.zeros((), device=self.device)

        index = torch.randint(0, self.num_old_classes, (batch_size,), device=self.device)
        proto_features = torch.stack(self._protos, dim=0)[index].to(self.device, non_blocking=True)
        proto_targets = index

        proto_logits = self.classifier(proto_features)["logits"][:, : self.num_seen_classes]
        proto_logits = self.semantic_aug(proto_logits, proto_targets)
        return F.cross_entropy(proto_logits / self.temp, proto_targets)

    def semantic_aug(self, proto_logits: torch.Tensor, proto_targets: torch.Tensor) -> torch.Tensor:
        weight = self.classifier.weight[: self.num_seen_classes]
        n, c, d = len(proto_targets), self.num_seen_classes, weight.shape[1]

        weight = weight.unsqueeze(0).expand(n, c, d)
        target_weight = torch.gather(weight, 1, proto_targets[:, None, None].expand(n, c, d))
        delta = weight - target_weight

        cov = torch.stack(self._covs, dim=0)[proto_targets].to(self.device, non_blocking=True)
        aug = torch.diagonal(delta @ cov @ delta.transpose(1, 2), dim1=1, dim2=2)
        return proto_logits + self.ratio * aug / 2

    @rank_zero_only
    def build_protos(self, dm: "HFDataModule"):
        if self.num_seen_classes <= self.num_old_classes:
            return

        self.eval()
        with torch.no_grad():
            features, targets = [], []
            for batch in tqdm(dm.train_dataloader(), desc=f"Building prototypes task {dm.get_current_task()}"):
                x, y = self.unpack_batch(batch)
                features.append(self.forward_layerwise(x.to(self.device))["features"])
                targets.append(y.to(self.device))

            features = torch.cat(features, dim=0)
            targets = torch.cat(targets, dim=0)

            for class_idx in range(self.num_old_classes, self.num_seen_classes):
                class_features = features[targets == class_idx]
                if len(class_features) == 0:
                    continue

                self._protos.append(class_features.mean(dim=0))

                if len(class_features) == 1:
                    dim = class_features.shape[1]
                    cov = torch.zeros(dim, dim, device=class_features.device, dtype=class_features.dtype)
                else:
                    centered = class_features - class_features.mean(dim=0, keepdim=True)
                    cov = centered.T @ centered / (len(class_features) - 1)
                self._covs.append(cov)
