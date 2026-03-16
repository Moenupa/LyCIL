from tqdm import tqdm
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from .base import BaseLearner


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

        # 每个元素对应一个类:
        # self._prototypes[class_idx] -> [num_rotations, feat_dim]
        self._prototypes: list[torch.Tensor] = []

        # 每个元素对应一个类:
        # self._prototype_radii[class_idx] -> [num_rotations]
        self._prototype_radii: list[torch.Tensor] = []

        self._radius: float = 0.0

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
        self.update_prototypes(self.trainer.datamodule)

    def rotate_batch(
            self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_rot = torch.stack(
            [torch.rot90(x, k, dims=(2, 3)) for k in range(self.num_rotations)],
            dim=1,
        )  # [B, R, C, H, W]
        x_rot = x_rot.flatten(0, 1)  # [B*R, C, H, W]

        y_rot = torch.stack(
            [y * self.num_rotations + k for k in range(self.num_rotations)],
            dim=1,
        ).flatten()  # [B*R]
        return x_rot, y_rot

    def prototype_loss(self, batch_size: int) -> torch.Tensor:
        if self.num_old_classes == 0 or not self._prototypes:
            return torch.zeros((), device=self.device)

        # [num_old_classes, R, D]
        proto_bank = torch.stack(self._prototypes[: self.num_old_classes], dim=0).to(self.device)

        # 随机采样 old class 和 rotation
        class_indices = torch.randint(
            low=0,
            high=self.num_old_classes,
            size=(batch_size,),
            device=self.device,
        )
        rot_indices = torch.randint(
            low=0,
            high=self.num_rotations,
            size=(batch_size,),
            device=self.device,
        )

        # [B, D]
        proto_features = proto_bank[class_indices, rot_indices]

        if self._radius > 0:
            proto_features = proto_features + torch.randn_like(proto_features) * self._radius

        # target = class_id * R + rot_id
        proto_targets = class_indices * self.num_rotations + rot_indices

        proto_logits = self.classifier(proto_features)["logits"]
        return F.cross_entropy(proto_logits / self.temp, proto_targets)

    @rank_zero_only
    def update_prototypes(self, dm) -> None:
        if self.num_seen_classes <= self.num_old_classes:
            return

        self.eval()
        with torch.no_grad():
            feats_all = []
            labels_all = []
            rots_all = []

            for batch in tqdm(
                    dm.train_dataloader(),
                    desc=f"Building prototypes task {dm.get_current_task()}",
            ):
                x, y = self.unpack_batch(batch)
                x = x.to(self.device)
                y = y.to(self.device)

                for rot_id in range(self.num_rotations):
                    x_rot = torch.rot90(x, rot_id, dims=(2, 3))
                    feats_rot = self.forward_layerwise(x_rot)["features"]  # [B, D]

                    feats_all.append(feats_rot)
                    labels_all.append(y)
                    rots_all.append(
                        torch.full_like(y, fill_value=rot_id, device=self.device)
                    )

            feats = torch.cat(feats_all, dim=0)     # [N*R, D]
            labels = torch.cat(labels_all, dim=0)   # [N*R]
            rot_ids = torch.cat(rots_all, dim=0)    # [N*R]

            new_prototypes = []
            new_radii = []

            for class_idx in range(self.num_old_classes, self.num_seen_classes):
                class_proto_list = []
                class_radius_list = []
                for rot_id in range(self.num_rotations):
                    mask = (labels == class_idx) & (rot_ids == rot_id)
                    class_rot_feats = feats[mask]


                    class_proto_list.append(class_rot_feats.mean(dim=0))

                    if len(class_rot_feats) > 1:
                        radius = class_rot_feats.var(dim=0, unbiased=True).mean()
                    else:
                        radius = torch.tensor(0.0, device=self.device)

                    class_radius_list.append(radius)

                # [R, D]
                class_protos = torch.stack(class_proto_list, dim=0)
                # [R]
                class_radii = torch.stack(class_radius_list, dim=0)

                new_prototypes.append(class_protos.detach().cpu())
                new_radii.append(class_radii.detach().cpu())

            self._prototypes.extend(new_prototypes)
            self._prototype_radii.extend(new_radii)

            if self._prototype_radii:
                all_radii = torch.cat([r.flatten() for r in self._prototype_radii], dim=0).float()
                self._radius = float(all_radii.mean().sqrt())

        self.train()