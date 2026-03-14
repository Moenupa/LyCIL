import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from tqdm import tqdm
from .base import BaseLearner
from ..data.hfmodule import HFDataModule


class SSRE(BaseLearner):
    def __init__(
        self,
        *args,
        temp: float = 1.0,
        lambda_fkd: float = 1.0,
        lambda_proto: float = 1.0,
        first_task_batch_size: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.temp = float(temp)
        self.lambda_fkd = float(lambda_fkd)
        self.lambda_proto = float(lambda_proto)
        self.first_task_batch_size = int(first_task_batch_size)

        self.task_size = 0
        self.num_old_classes = 0

        self._protos: list[torch.Tensor] = []
        self._adapter_prepared = False

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

        self.expand_head(task_size)

        self.task_size = task_size
        self.num_old_classes = self.num_seen_classes or 0
        self.num_seen_classes = dm.num_seen_classes
        self._adapter_prepared = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_layerwise(x)["logits"][:, : self.num_seen_classes]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = self.unpack_batch(batch)

        outs = self.forward_layerwise(x)
        logits = outs["logits"][:, : self.num_seen_classes]
        features = outs["features"]

        loss_ce = F.cross_entropy(logits / self.temp, y)
        loss = loss_ce
        loss_distill = None
        loss_proto = None
        mask_mean = None

        if self.task_id > 0:
            with torch.no_grad():
                old_features = self.old_self.forward_layerwise(x)["features"]

            mask = self.similarity_mask(features)
            loss_ce = (F.cross_entropy(logits / self.temp, y, reduction="none") * (1.0 - mask)).mean()
            loss_distill = torch.sum(torch.norm(features - old_features, p=2, dim=1) * mask)
            loss_proto = self.prototype_loss(x.shape[0])

            loss = loss_ce + self.lambda_fkd * loss_distill + self.lambda_proto * loss_proto
            mask_mean = mask.mean()

        self.log_dict(
            {
                "train/loss": loss,
                "train/ce": loss_ce,
                "train/loss_distill": loss_distill or 0.0,
                "train/loss_proto": loss_proto or 0.0,
                "train/mask_mean": mask_mean or 0.0,
                "train/x_mean": x.detach().float().mean(),
                "train/x_var": x.detach().float().var(unbiased=False),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_fit_start(self) -> None:
        self.prepare_adapters()
        return super().on_fit_start()

    def on_train_end(self):
        self.build_protos(self.trainer.datamodule)
        self.compress_adapters()

    def configure_optimizers(self):
        self.prepare_adapters()
        return super().configure_optimizers()

    def prepare_adapters(self) -> None:
        if self._adapter_prepared:
            return

        convnet = self.convnet
        if self.task_id and self.task_id > 0:
            for p in convnet.parameters():
                p.requires_grad = True
            for name, p in convnet.named_parameters():
                if "adapter" not in name:
                    p.requires_grad = False
            convnet.switch("parallel_adapters")

        self._adapter_prepared = True

    def compress_adapters(self) -> None:
        if not self.task_id or self.task_id <= 0:
            return

        state_dict = self.state_dict()
        for name, value in list(state_dict.items()):
            if "adapter" not in name:
                continue

            conv_name = name.replace("adapter", "conv")
            if conv_name not in state_dict:
                continue

            if name.endswith("weight"):
                state_dict[conv_name] = state_dict[conv_name] + F.pad(value, [1, 1, 1, 1], "constant", 0)
                state_dict[name] = torch.zeros_like(value)
            elif name.endswith("bias"):
                state_dict[conv_name] = state_dict[conv_name] + value
                state_dict[name] = torch.zeros_like(value)

        self.load_state_dict(state_dict)
        self.convnet.switch("normal")

    def similarity_mask(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_old_classes == 0 or not self._protos:
            return torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)

        protos = torch.stack(self._protos[: self.num_old_classes], dim=0).to(features.device, non_blocking=True)
        feat_norm = F.normalize(features, p=2, dim=1, eps=1e-12)
        proto_norm = F.normalize(protos, p=2, dim=1, eps=1e-12)
        return (feat_norm @ proto_norm.T).max(dim=1).values.detach()

    def prototype_loss(self, batch_size: int) -> torch.Tensor:
        if self.num_old_classes == 0 or not self._protos:
            return torch.zeros((), device=self.device)

        index = torch.randint(0, self.num_old_classes, (batch_size,), device=self.device)
        proto_features = torch.stack(self._protos[: self.num_old_classes], dim=0)[index]
        proto_targets = index

        proto_logits = self.classifier(proto_features)["logits"][:, : self.num_seen_classes]
        return F.cross_entropy(proto_logits / self.temp, proto_targets)

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
