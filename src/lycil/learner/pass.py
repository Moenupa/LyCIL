import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseLearner


class PASS(BaseLearner):

    def __init__(
        self,
        *args,
        temp: float = 1.0,
        lambda_fkd: float = 1.0,
        lambda_proto: float = 1.0,
        num_rotations: int = 4,
        proto_batch_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.temp = float(temp)
        self.lambda_fkd = float(lambda_fkd)
        self.lambda_proto = float(lambda_proto)
        self.num_rotations = int(num_rotations)
        self.proto_batch_size = proto_batch_size

        self._prototypes: list[torch.Tensor] = []
        self._prototype_radii: list[float] = []
        self._radius: float = 0.0

    @property
    def classifier_num_classes(self) -> int:
        """Compatibility hook for bases that query classifier output size."""
        return self.num_seen_classes * self.num_rotations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class-level logits for evaluation/inference."""
        raw_logits = self._forward_raw(x)["logits"]
        return self._collapse_logits(raw_logits)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = self.unpack_batch(batch)
        x_rot, y_rot = self._expand_batch(x, y)

        outputs = self._forward_raw(x_rot)
        logits, features = outputs["logits"], outputs["features"]

        loss_ce = F.cross_entropy(logits / self.temp, y_rot)
        loss = loss_ce

        loss_feature_distill = None
        loss_proto = None

        if self.task_id > 0:
            with torch.no_grad():
                old_outputs = self.old_self._forward_raw(x_rot)
                old_features = old_outputs["features"]

            loss_feature_distill = self.lambda_fkd * torch.dist(
                features, old_features, p=2
            )
            loss = loss + loss_feature_distill

            loss_proto = self._prototype_loss(batch_size=x.size(0))
            loss = loss + loss_proto

        self.log_dict(
            {
                "train/loss": loss,
                "train/ce": loss_ce,
                "train/feature_distill": loss_feature_distill or 0.0,
                "train/proto": loss_proto or 0.0,
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
        self.update_memory(datamodule)
        self.update_prototypes(datamodule)

    def update_prototypes(self, datamodule) -> None:
        """Append task-local class prototypes and update their shared radius."""
        if not hasattr(datamodule, "get_dataset"):
            raise AttributeError(
                "PASS expects the datamodule to expose `get_dataset(...)` for "
                "prototype extraction."
            )

        start_class = self.num_old_classes
        end_class = self.num_seen_classes
        if end_class <= start_class:
            return

        device = self.device
        was_training = self.training
        self.eval()

        new_prototypes: list[torch.Tensor] = []
        new_radii: list[float] = []

        with torch.no_grad():
            for class_idx in range(start_class, end_class):
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
                    x, _ = self._unpack_any_batch(batch)
                    outputs = self._forward_raw(x.to(device, non_blocking=True))
                    feats.append(outputs["features"].detach().cpu())

                features = torch.cat(feats, dim=0)
                new_prototypes.append(features.mean(dim=0))
                if features.size(0) > 1:
                    new_radii.append(float(features.var(dim=0, unbiased=True).mean()))
                else:
                    new_radii.append(0.0)

        if was_training:
            self.train()

        self._prototypes.extend(new_prototypes)
        self._prototype_radii.extend(new_radii)
        if self._prototype_radii:
            radius = torch.tensor(self._prototype_radii, dtype=torch.float32).mean()
            self._radius = float(radius.sqrt().item())

    def _forward_raw(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Bypass PASS logit collapsing and expose raw rotation-expanded outputs."""
        return BaseLearner.forward_layerwise(self, x)

    def _expand_batch(
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

    def _unpack_any_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return self.unpack_batch(batch)
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            return batch[1], batch[2]
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch
        raise TypeError(f"Unsupported batch type for PASS prototype extraction: {type(batch)!r}")

    def _prototype_loss(self, batch_size: int) -> torch.Tensor:
        if self.num_old_classes == 0 or not self._prototypes:
            return torch.zeros((), device=self.device)

        proto_batch_size = self.proto_batch_size or batch_size
        proto_bank = torch.stack(self._prototypes[: self.num_old_classes], dim=0)

        indices = torch.randint(
            low=0,
            high=self.num_old_classes,
            size=(proto_batch_size,),
            device=proto_bank.device,
        )
        proto_features = proto_bank[indices]
        if self._radius > 0:
            proto_features = proto_features + torch.randn_like(proto_features) * self._radius

        proto_features = proto_features.to(self.device, non_blocking=True)
        proto_targets = (indices * self.num_rotations).to(self.device, non_blocking=True)

        head = self._get_classifier_head()
        head_outputs = head(proto_features)
        proto_logits = (
            head_outputs["logits"]
            if isinstance(head_outputs, dict)
            else head_outputs
        )
        return self.lambda_proto * F.cross_entropy(
            proto_logits / self.temp,
            proto_targets,
        )

    def _get_classifier_head(self):
        for attr in ("head", "fc", "classifier"):
            if hasattr(self, attr):
                return getattr(self, attr)
        raise AttributeError(
            "PASS could not find a classifier head. Expected one of: `head`, `fc`, `classifier`."
        )


