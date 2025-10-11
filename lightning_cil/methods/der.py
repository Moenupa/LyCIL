import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseIncremental

class _Adapter(nn.Module):
    """Linear adapter that expands representation per task."""
    def __init__(self, d: int, rank: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(d, rank)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(rank, d)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class DER(BaseIncremental):
    """Dynamically Expandable Representation with per-task adapters + KD."""
    def __init__(
        self, *args,
        adapter_rank: int = 256,
        lambda_kd: float = 1.0,
        kd_T: float = 2.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.adapters = nn.ModuleList()
        self.adapter_rank = int(adapter_rank)
        self.lambda_kd = float(lambda_kd)
        self.kd_T = float(kd_T)
        self._num_tasks_seen = 0

    def set_task_info(self, current_classes, seen_classes):
        super().set_task_info(current_classes, seen_classes)
        # Create a new adapter for this task
        if len(self.adapters) < (self._num_tasks_seen + 1):
            self.adapters.append(_Adapter(self.feature_dim, self.adapter_rank))
            # freeze previous adapters
            for i, a in enumerate(self.adapters[:-1]):
                for p in a.parameters(): p.requires_grad = False
            self._num_tasks_seen += 1

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        # Sum base with all adapters (old frozen, current trainable)
        for a in self.adapters:
            f = f + a(f)
        return f

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss_ce = F.cross_entropy(logits, y)

        loss_kd = torch.tensor(0.0, device=x.device)
        if self.prev_model is not None and len(self.seen_classes) > 0:
            with torch.no_grad():
                prev_logits = self.prev_model(x)
            old_idx = torch.tensor(sorted(self.seen_classes), device=x.device, dtype=torch.long)
            old_idx = old_idx[old_idx < logits.size(1)]
            if old_idx.numel() > 0:
                T = self.kd_T
                p = F.log_softmax(logits.index_select(1, old_idx)/T, dim=1)
                q = F.softmax(prev_logits.index_select(1, old_idx)/T, dim=1)
                loss_kd = F.kl_div(p, q, reduction="batchmean") * (T*T)

        loss = loss_ce + self.lambda_kd * loss_kd
        self.log_dict({"train/loss": loss, "train/ce": loss_ce, "train/kd": loss_kd},
                      prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def update_memory(self, datamodule, device: torch.device):
        # DER typically uses exemplars; reuse herding
        per_class = self.buffer.exemplars_per_class(num_classes_seen=len(self.seen_classes))
        self.buffer.reduce_exemplars(per_class)
        for c in self.current_classes:
            loader = datamodule.build_class_loader(c, train=True, batch_size=128, shuffle=False, num_workers=datamodule.num_workers)
            self.buffer.build_exemplars_herding(self, self.feature_extractor, loader, c, per_class, device)
