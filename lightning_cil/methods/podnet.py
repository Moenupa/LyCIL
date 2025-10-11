import torch
import torch.nn.functional as F
from .base import BaseIncremental
from utils.pod import pod_loss

class PODNet(BaseIncremental):
    """PODNet: pooled outputs distillation at multiple backbone blocks."""
    def __init__(
        self, *args,
        pod_lambda: float = 1.0,
        kd_T: float = 2.0,
        lambda_kd: float = 1.0,
        **kwargs
    ):
        kwargs.setdefault("head", "cosine")
        super().__init__(*args, **kwargs)
        self.pod_lambda = float(pod_lambda)
        self.kd_T = float(kd_T)
        self.lambda_kd = float(lambda_kd)

    def _forward_feats(self, x):
        feats = self.backbone.forward_feats(x)
        g = self.classifier(self.backbone.pool(feats["l4"]).flatten(1))
        return g, feats

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, feats = self._forward_feats(x)
        loss_ce = F.cross_entropy(logits, y)

        loss_pod = torch.tensor(0.0, device=x.device)
        loss_kd = torch.tensor(0.0, device=x.device)

        if self.prev_model is not None and len(self.seen_classes) > 0:
            with torch.no_grad():
                prev_logits, prev_feats = self.prev_model._forward_feats(x)
            loss_pod = pod_loss(feats, prev_feats)

            # KD on old classes
            old_idx = torch.tensor(sorted(self.seen_classes), device=x.device, dtype=torch.long)
            old_idx = old_idx[old_idx < logits.size(1)]
            if old_idx.numel() > 0:
                T = self.kd_T
                p = F.log_softmax(logits.index_select(1, old_idx)/T, dim=1)
                q = F.softmax(prev_logits.index_select(1, old_idx)/T, dim=1)
                loss_kd = F.kl_div(p, q, reduction="batchmean") * (T*T)

        loss = loss_ce + self.pod_lambda * loss_pod + self.lambda_kd * loss_kd
        self.log_dict(
            {"train/loss": loss, "train/ce": loss_ce, "train/pod": loss_pod, "train/kd": loss_kd},
            prog_bar=True, on_epoch=True, sync_dist=True
        )
        return loss

    @torch.no_grad()
    def update_memory(self, datamodule, device: torch.device):
        # Use standard herding like LUCIR/Replay
        per_class = self.buffer.exemplars_per_class(num_classes_seen=len(self.seen_classes))
        self.buffer.reduce_exemplars(per_class)
        for c in self.current_classes:
            loader = datamodule.build_class_loader(c, train=True, batch_size=128, shuffle=False, num_workers=datamodule.num_workers)
            self.buffer.build_exemplars_herding(self, self.feature_extractor, loader, c, per_class, device)
