from tqdm import tqdm
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from .base import BaseLearner
from ..constants import _X_COLUMN_NAME, _Y_COLUMN_NAME


class PASS(BaseLearner):

    def __init__(
            self,
            *args,
            temp: float = 1.0,
            lambda_kd: float = 1.0,
            lambda_proto: float = 1.0,
            num_rotations: int = 4,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.temp = float(temp)
        self.lambda_kd = float(lambda_kd)
        self.lambda_proto = float(lambda_proto)
        self.num_rotations = int(num_rotations)

        self._prototypes: list[torch.Tensor] = []
        self._prototype_radii: list[float] = []
        self._mean_radius: float = 0.0

    def sync_with_datamodule(self, dm: "HFDataModule"):
        dm_task_id = dm.get_current_task()
        if self.task_id is not None and dm_task_id == self.task_id:
            return

        self.task_id = dm_task_id

        incoming_expansion = dm.num_seen_classes - (self.num_seen_classes or 0)
        if incoming_expansion <= 0:
            raise RuntimeError(
                f"Expect an incoming expansion, got {incoming_expansion} new classes. "
                + f"Data has {dm.num_seen_classes} seen classes, "
                + f"but Model has {self.num_seen_classes} seen classes. "
                + "Ensure that `sync_with_datamodule()` is called after datamodule updates."
            )

        # expand head with num_rotations
        self.expand_head(incoming_expansion * self.num_rotations)

        self.num_old_classes = self.num_seen_classes or 0
        self.num_seen_classes = dm.num_seen_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class-level logits for evaluation/inference."""
        logits_with_rotate = self.forward_layerwise(x)["logits"]
        logits = logits_with_rotate[:, :: self.num_rotations]
        return logits

    def training_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = self.unpack_batch(batch)
        x_rot, y_rot = self.rotate_batch(x, y)

        outputs = self.forward_layerwise(x_rot)
        logits, features = outputs["logits"], outputs["features"]

        loss_ce = F.cross_entropy(logits / self.temp, y_rot)
        loss = loss_ce

        loss_kd = None
        loss_proto = None

        if self.task_id > 0:
            with torch.no_grad():
                old_outputs = self.old_self.forward_layerwise(x_rot)
                old_features = old_outputs["features"]

            loss_kd = torch.dist(features, old_features, p=2)
            loss_proto = self.prototype_loss(batch_size=x.shape[0])

            loss = loss + self.lambda_kd * loss_kd + self.lambda_proto * loss_proto

        self.log_dict(
            {
                "train/loss": loss,
                "train/loss_ce": loss_ce,
                "train/loss_kd": loss_kd or 0.0,
                "train/loss_proto": loss_proto or 0.0,
                "train/proto_radius_mean": float(self._mean_radius),
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
        """Update rehearsal memory and refresh old-class prototypes."""
        self.update_prototypes(self.trainer.datamodule)

    def rotate_batch(
            self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_rot = torch.stack(
            [torch.rot90(x, k, dims=(2, 3)) for k in range(self.num_rotations)],
            dim=1,
        )
        x_rot = x_rot.flatten(0, 1)

        y_rot = torch.stack(
            [y * self.num_rotations + k for k in range(self.num_rotations)],
            dim=1,
        ).flatten()
        return x_rot, y_rot

    def prototype_loss(self, batch_size: int) -> torch.Tensor:
        """
        Prototype replay with class-specific radii.

        注意：
        - 这里回放的是“类别原型”，因此采用 class-level CE 更合理。
        - 如果你想做 rotation-level prototype replay，建议把原型存成 (class, rotation)。
        """
        if self.num_old_classes == 0 or len(self._prototypes) < self.num_old_classes:
            return torch.zeros((), device=self.device)

        proto_bank = torch.stack(
            self._prototypes[: self.num_old_classes], dim=0
        ).to(self.device)

        radius_bank = torch.as_tensor(
            self._prototype_radii[: self.num_old_classes],
            device=self.device,
            dtype=proto_bank.dtype,
        )

        cls_idx = torch.randint(
            low=0,
            high=self.num_old_classes,
            size=(batch_size,),
            device=self.device,
        )

        proto_features = proto_bank[cls_idx]                      # [B, D]
        sampled_radii = radius_bank[cls_idx].unsqueeze(1)        # [B, 1]

        if torch.any(sampled_radii > 0):
            proto_features = proto_features + torch.randn_like(proto_features) * sampled_radii

        # class-level replay
        proto_logits_with_rotate = self.classifier(proto_features)["logits"]
        proto_logits = proto_logits_with_rotate[:, :: self.num_rotations]  # [B, num_seen_classes]

        return F.cross_entropy(proto_logits / self.temp, cls_idx)

    def update_prototypes(self, dm) -> None:
        if self.num_seen_classes <= self.num_old_classes:
            return

        self.eval()
        with torch.no_grad():
            feats, labels = [], []
            for batch in tqdm(
                    dm.train_dataloader(),
                    desc=f"Building prototypes task {dm.get_current_task()}",
            ):
                x, y = self.unpack_batch(batch)
                feats.append(self.forward_layerwise(x.to(self.device))["features"])
                labels.append(y.to(self.device))

            feats = torch.cat(feats, dim=0)
            labels = torch.cat(labels, dim=0)

            new_prototypes, new_radii = [], []
            for class_idx in range(self.num_old_classes, self.num_seen_classes):
                class_feats = feats[labels == class_idx]
                if class_feats.size(0) == 0:
                    continue

                # prototype
                class_proto = class_feats.mean(dim=0)
                new_prototypes.append(class_proto.detach().cpu())

                # radius: sqrt(mean(var))
                if class_feats.size(0) > 1:
                    class_radius = class_feats.var(dim=0, unbiased=True).mean().sqrt()
                    new_radii.append(float(class_radius.item()))
                else:
                    new_radii.append(0.0)

            self._prototypes.extend(new_prototypes)
            self._prototype_radii.extend(new_radii)

            if self._prototype_radii:
                self._mean_radius = float(torch.tensor(self._prototype_radii).mean().item())
            else:
                self._mean_radius = 0.0

        self.train()