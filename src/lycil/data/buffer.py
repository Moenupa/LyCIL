from typing import Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class ExemplarDataset(Dataset):
    """Dataset wrapper around exemplars stored as tensors (C,H,W) and labels."""

    def __init__(self, items: list[tuple[torch.Tensor, int]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, label = self.items[idx]
        return img, label


class ExemplarBuffer:
    """Fixed-size memory with per-class lists and herding selection.

    Stores exemplar images as float tensors already normalized to dataset stats.
    """

    def __init__(self, mem_size: int = 2000):
        self.mem_size = int(mem_size)
        self.storage: dict[int, list[torch.Tensor]] = {}
        self.class_means: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self.storage.values())

    def classes(self) -> list[int]:
        return sorted(list(self.storage.keys()))

    def exemplars_per_class(self, num_classes_seen: int) -> int:
        if num_classes_seen <= 0:
            return 0
        return self.mem_size // num_classes_seen

    @torch.no_grad()
    def build_exemplars_herding(
        self,
        model,
        feature_extractor,
        dataloader,
        class_id: int,
        m: int,
        device: torch.device,
    ):
        """Select m exemplars for class_id by herding using current model features."""
        model.eval()
        feats = []
        imgs = []
        for x, y in dataloader:
            mask = y == class_id
            if mask.sum() == 0:
                continue
            x = x[mask].to(device, non_blocking=True)
            f = feature_extractor(x).detach()  # (n,d)
            f = F.normalize(f, dim=1)
            feats.append(f.cpu())
            imgs.append(x.cpu())
        if not feats:
            return
        feats = torch.cat(feats, dim=0)  # (N,d)
        imgs = torch.cat(imgs, dim=0)  # (N,C,H,W)

        mu = feats.mean(dim=0, keepdim=True)  # (1,d)
        selected = []
        selected_idx = []
        running = torch.zeros_like(mu)
        taken = torch.zeros(feats.size(0), dtype=torch.bool)

        for k in range(min(m, feats.size(0))):
            # pick argmin of ||mu - (running + f_i)/(k+1)||
            remain = (~taken).nonzero(as_tuple=False).squeeze(1)
            if remain.numel() == 0:
                break
            candidates = feats[remain]
            delta = mu - (running + candidates) / float(k + 1)
            idx = torch.argmin(torch.norm(delta, dim=1))
            choice = remain[idx].item()
            taken[choice] = True
            selected_idx.append(choice)
            running = running + feats[choice : choice + 1]
            selected.append(imgs[choice])

        self.storage[class_id] = [img.clone() for img in selected]
        self.class_means[class_id] = mu.squeeze(0)
        model.train()

    def reduce_exemplars(self, per_class_quota: int):
        """Downsize exemplar lists after new classes arrive."""
        for c in list(self.storage.keys()):
            self.storage[c] = self.storage[c][:per_class_quota]

    def add_exemplars(self, class_id: int, images: Iterable[torch.Tensor]):
        """Append preselected images (already normalized)."""
        lst = self.storage.setdefault(class_id, [])
        for img in images:
            lst.append(img.clone())

    def make_dataset(self) -> Dataset:
        items = []
        for c, imgs in self.storage.items():
            for t in imgs:
                items.append((t, c))
        return ExemplarDataset(items)

    @torch.no_grad()
    def compute_nme_classifier(
        self, model, feature_extractor, device: torch.device
    ) -> dict[int, torch.Tensor]:
        """Compute mean feature per class from exemplars (for iCaRL NME inference)."""
        nme = {}
        for c, imgs in self.storage.items():
            if not imgs:
                continue
            x = torch.stack(imgs).to(device)
            f = feature_extractor(x)
            f = F.normalize(f, dim=1)
            nme[c] = f.mean(dim=0).detach().cpu()
        return nme
