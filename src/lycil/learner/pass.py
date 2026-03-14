import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .base import BaseLearner


class PASS(BaseLearner):

    def __init__(
            self,
            *args,
            temp: float = 1.0,
            lambda_fkd: float = 1.0,
            lambda_proto: float = 1.0,
            num_rotations: int = 4,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.temp = float(temp)
        self.lambda_fkd = float(lambda_fkd)
        self.lambda_proto = float(lambda_proto)
        self.num_rotations = int(num_rotations)

        self._prototypes: list[torch.Tensor] = []
        self._prototype_radii: list[float] = []
        self._radius: float = 0.0

    def training_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = self.unpack_batch(batch)
        x_rot, y_rot = self.rotate_batch(x, y)

        outputs = self.forward_layerwise(x_rot)
        logits, features = outputs["logits"], outputs["features"]

        loss_ce = F.cross_entropy(logits / self.temp, y_rot)
        loss = loss_ce

        loss_distill = None
        loss_proto = None

        if self.task_id > 0:
            with torch.no_grad():
                old_outputs = self.old_self.forward_layerwise(x_rot)
                old_features = old_outputs["features"]

            loss_distill = torch.dist(features, old_features, p=2)
            loss_proto = self.prototype_loss(batch_size=x.shape[0])
            loss = loss + self.lambda_fkd * loss_distill + loss_proto

        self.log_dict(
            {
                "train/loss": loss,
                "train/ce": loss_ce,
                "train/loss_distill": loss_distill or 0.0,
                "train/loss_proto": loss_proto or 0.0,
                "train/proto_radius": float(self._radius),
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
        datamodule = self.trainer.datamodule  # ty: ignore[unresolved-attribute]
        self.update_prototypes(datamodule)

    @rank_zero_only
    def update_prototypes(self, datamodule) -> None:
        """Append task-local class prototypes and update their shared radius."""
        if self.num_seen_classes <= self.num_old_classes:
            return
        new_prototypes: list[torch.Tensor] = []
        new_radii: list[float] = []
        self.eval()
        with torch.no_grad():
            for class_idx in range(self.num_old_classes, self.num_seen_classes):
                _, _, dataset = datamodule.get_dataset(
                    range(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=getattr(datamodule, "batch_size", 128),
                    shuffle=False,
                    num_workers=getattr(datamodule, "num_workers", 4),
                    pin_memory=True,
                )

                feats = []
                for batch in loader:
                    x, _ = self.unpack_batch(batch)
                    outputs = self.forward_layerwise(x.to(self.device, non_blocking=True))
                    feats.append(outputs["features"].detach().cpu())

                features = torch.cat(feats, dim=0)
                new_prototypes.append(features.mean(dim=0))
                if features.size(0) > 1:
                    new_radii.append(float(features.var(dim=0, unbiased=True).mean()))
                else:
                    new_radii.append(0.0)


        self._prototypes.extend(new_prototypes)
        self._prototype_radii.extend(new_radii)
        if self._prototype_radii:
            radius = torch.tensor(self._prototype_radii, dtype=torch.float32).mean()
            self._radius = float(radius.sqrt().item())

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

    def _collapse_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, :: self.num_rotations]

    # TODO: concate features with proto features
    def prototype_loss(self, batch_size: int) -> torch.Tensor:
        if self.num_old_classes == 0 or not self._prototypes:
            return torch.zeros((), device=self.device)

        proto_bank = torch.stack(self._prototypes[: self.num_old_classes], dim=0)

        indices = torch.randint(
            low=0,
            high=self.num_old_classes,
            size=(batch_size,),
            device=self.device,
        )
        proto_features = proto_bank[indices]
        if self._radius > 0:
            proto_features = proto_features + torch.randn_like(proto_features) * self._radius

        proto_features = proto_features.to(self.device, non_blocking=True)
        proto_targets = (indices * self.num_rotations).to(self.device, non_blocking=True)

        proto_logits = self.classifier(proto_features)["logits"]
        return self.lambda_proto * F.cross_entropy(
            proto_logits / self.temp,
            proto_targets,
        )
