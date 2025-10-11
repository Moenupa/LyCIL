from typing import Dict, List, Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from .base import BaseIncremental

class BiCBias(nn.Module):
    """Bias Correction layer: y_old unchanged; y_new -> alpha*y_new + beta."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, logits, new_idx: torch.Tensor):
        out = logits.clone()
        out[:, new_idx] = self.alpha * out[:, new_idx] + self.beta
        return out

class BiC(BaseIncremental):
    """BiC (CVPR'19): KD + CE training, then bias correction on a held-out exemplar val split."""
    def __init__(
        self, *args,
        kd_T: float = 2.0,
        lambda_kd: float = 1.0,
        bic_val_fraction: float = 0.1,
        bic_epochs: int = 20,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.kd_T = float(kd_T)
        self.lambda_kd = float(lambda_kd)
        self.bic_val_fraction = float(bic_val_fraction)
        self.bic_epochs = int(bic_epochs)
        self.bias_layers: List[Tuple[List[int], BiCBias]] = []  # list of (new_class_ids, bias_layer)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss_ce = F.cross_entropy(logits, y)

        loss_kd = torch.tensor(0.0, device=x.device)
        if self.prev_model is not None and len(self.seen_classes) > 0:
            old_idx = torch.tensor(sorted(self.seen_classes), device=x.device, dtype=torch.long)
            old_idx = old_idx[old_idx < logits.size(1)]
            if old_idx.numel() > 0:
                with torch.no_grad():
                    prev_logits = self.prev_model(x)
                T = self.kd_T
                p = F.log_softmax(logits.index_select(1, old_idx) / T, dim=1)
                q = F.softmax(prev_logits.index_select(1, old_idx) / T, dim=1)
                loss_kd = F.kl_div(p, q, reduction="batchmean") * (T*T)

        loss = loss_ce + self.lambda_kd * loss_kd
        self.log_dict({"train/loss": loss, "train/ce": loss_ce, "train/kd": loss_kd},
                      prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def update_memory(self, datamodule, device: torch.device):
        """Update exemplars and fit bias correction on a held-out split."""
        per_class = self.buffer.exemplars_per_class(num_classes_seen=len(self.seen_classes))
        self.buffer.reduce_exemplars(per_class)
        for c in self.current_classes:
            loader = datamodule.build_class_loader(c, train=True, batch_size=128, shuffle=False, num_workers=datamodule.num_workers)
            self.buffer.build_exemplars_herding(self, self.feature_extractor, loader, c, per_class, device)

        # Train bias correction layer for new classes
        if len(self.current_classes) == 0:
            return
        # Split exemplars into train/val
        all_items = []
        for c in self.seen_classes:
            for img in self.buffer.storage.get(c, []):
                all_items.append((img.clone(), c))
        if not all_items:
            return
        random.shuffle(all_items)
        n_val = max(1, int(self.bic_val_fraction * len(all_items)))
        val_items = all_items[:n_val]
        train_items = all_items[n_val:]

        def make_loader(items):
            class _MemDataset(torch.utils.data.Dataset):
                def __init__(self, items): self.items = items
                def __len__(self): return len(self.items)
                def __getitem__(self, i): 
                    x, y = self.items[i]
                    return x, torch.tensor(y, dtype=torch.long)
            if len(items) == 0:
                return None
            ds = _MemDataset(items)
            return torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

        tr_loader = make_loader(train_items)
        va_loader = make_loader(val_items)
        if tr_loader is None or va_loader is None:
            return

        bias = BiCBias().to(device)
        optim = torch.optim.SGD(bias.parameters(), lr=0.01, momentum=0.9)

        new_idx = torch.tensor(sorted(self.current_classes), device=device, dtype=torch.long)

        # Freeze model; train bias on top using exemplars
        self.eval()
        for epoch in range(self.bic_epochs):
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.no_grad():
                    logits = self(xb)
                logits_bc = bias(logits, new_idx)
                loss = F.cross_entropy(logits_bc, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()

        # Attach layer (keep as a list per step)
        self.bias_layers.append((list(new_idx.tolist()), bias))

    def forward(self, x):
        logits = super().forward(x)
        # apply bias corrections sequentially
        if len(self.bias_layers) > 0:
            for cls_ids, bias in self.bias_layers:
                idx = torch.tensor(cls_ids, device=logits.device, dtype=torch.long)
                logits = bias(logits, idx)
        return logits
