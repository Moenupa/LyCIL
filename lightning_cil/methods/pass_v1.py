import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseIncremental

def nt_xent(z1, z2, T=0.2):
    """NT-Xent contrastive loss using cosine similarity."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    sim = z1 @ z2.t() / T
    targets = torch.arange(B, device=z1.device)
    return F.cross_entropy(sim, targets)

class PASSV1(BaseIncremental):
    """PASS: prototype augmentation + self-supervision (contrastive)."""
    def __init__(
        self, *args,
        ssl_lambda: float = 0.2,
        ssl_T: float = 0.2,
        proto_aug_lambda: float = 0.2,
        **kwargs
    ):
        kwargs.setdefault("head", "cosine")
        super().__init__(*args, **kwargs)
        self.ssl_lambda = float(ssl_lambda)
        self.ssl_T = float(ssl_T)
        self.proto_aug_lambda = float(proto_aug_lambda)
        self.proj = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(inplace=True),
                                  nn.Linear(self.feature_dim, self.feature_dim))

    def training_step(self, batch, batch_idx):
        x, y = batch
        f = self.feature_extractor(x)
        logits = self.classifier(f)
        loss_ce = F.cross_entropy(logits, y)

        # Self-supervision via two stochastic passes (dropout acts as augmentation)
        f2 = self.feature_extractor(x)  # different dropout statistics if present
        z1 = self.proj(f)
        z2 = self.proj(f2)
        loss_ssl = nt_xent(z1, z2, T=self.ssl_T)

        # Prototype augmentation: mix features towards class prototypes from exemplars (if exist)
        loss_proto = torch.tensor(0.0, device=x.device)
        if len(self.buffer) > 0 and hasattr(self.buffer, "class_means"):
            with torch.no_grad():
                # build normalized prototype table from exemplars
                proto = self.buffer.compute_nme_classifier(self, self.feature_extractor, x.device)
            if proto:
                class_ids = sorted(proto.keys())
                P = torch.stack([proto[c] for c in class_ids]).to(x.device)  # (C,d)
                P = F.normalize(P, dim=1)
                # select prototype for each y
                idx = torch.tensor([class_ids.index(int(t.item())) for t in y], device=x.device)
                p_selected = P.index_select(0, idx)
                f_mix = F.normalize(0.7 * F.normalize(f, dim=1) + 0.3 * p_selected, dim=1)
                # encourage classifier to predict y on mixed features
                logits_mix = self.classifier(f_mix * f.norm(dim=1, keepdim=True))
                loss_proto = F.cross_entropy(logits_mix, y)

        loss = loss_ce + self.ssl_lambda * loss_ssl + self.proto_aug_lambda * loss_proto
        self.log_dict({"train/loss": loss, "train/ce": loss_ce, "train/ssl": loss_ssl, "train/proto": loss_proto},
                      prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def update_memory(self, datamodule, device: torch.device):
        # Use standard herding
        per_class = self.buffer.exemplars_per_class(num_classes_seen=len(self.seen_classes))
        self.buffer.reduce_exemplars(per_class)
        for c in self.current_classes:
            loader = datamodule.build_class_loader(c, train=True, batch_size=128, shuffle=False, num_workers=datamodule.num_workers)
            self.buffer.build_exemplars_herding(self, self.feature_extractor, loader, c, per_class, device)
