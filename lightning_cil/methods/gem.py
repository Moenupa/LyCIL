from typing import Dict, List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from .base import BaseIncremental
from utils.grad_utils import pack_grads, apply_grad_vector, zero_grads
from utils.gem_qp import project_to_cone

class GEM(BaseIncremental):
    """GEM (Lopez-Paz & Ranzato, NeurIPS 2017) with per-task episodic memory and gradient projection."""
    def __init__(self, *args, gem_memory_per_class: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False  # manual optimization to project gradients
        # track class->task mapping for per-task constraints
        self.class2task: Dict[int, int] = {}
        self.gem_memory_per_class = int(gem_memory_per_class)

    def set_task_info(self, current_classes: List[int], seen_classes: List[int]):
        super().set_task_info(current_classes, seen_classes)
        # assign current classes to current task id (inferred from len of seen before adding current)
        current_task = max([self.class2task.get(c, -1) for c in seen_classes], default=-1) + 1
        for c in current_classes:
            self.class2task[c] = current_task

    def _sample_memory_batch_per_task(self, datamodule, batch_size: int = 64):
        """Build a dict task_id -> (x,y) from exemplar buffer for each past task."""
        if len(self.buffer) == 0:
            return {}
        items = []
        for c, imgs in self.buffer.storage.items():
            for img in imgs:
                items.append((img.clone(), c))
        if not items:
            return {}
        # group by task
        import random
        by_task = {}
        for img, c in items:
            t = self.class2task.get(int(c), 0)
            by_task.setdefault(t, []).append((img, int(c)))
        batches = {}
        device = self.device
        for t, lst in by_task.items():
            if any(c in self.current_classes for _, c in lst):
                # only use strictly past tasks
                continue
            random.shuffle(lst)
            take = min(batch_size, len(lst))
            xs = torch.stack([lst[i][0] for i in range(take)]).to(device)
            ys = torch.tensor([lst[i][1] for i in range(take)], device=device, dtype=torch.long)
            batches[t] = (xs, ys)
        return batches

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        g_cur = pack_grads(self)

        # Build G (k x p) from past task memory
        mem_batches = self._sample_memory_batch_per_task(self.trainer.datamodule, batch_size=64)
        G_rows = []
        if mem_batches:
            for t, (mx, my) in sorted(mem_batches.items()):
                zero_grads(self)
                mlogits = self(mx)
                mloss = F.cross_entropy(mlogits, my)
                mloss.backward()
                g_t = pack_grads(self)
                G_rows.append(g_t)
        if G_rows:
            G = torch.stack(G_rows, dim=0)  # (k,p)
            # If any constraint violated, project
            violate = (G @ g_cur) < 0
            if violate.any():
                g_proj = project_to_cone(g_cur, G[violate])
                zero_grads(self)
                apply_grad_vector(self, g_proj)
        # step
        opt.step()
        self.log("train/loss", loss.detach(), prog_bar=True, on_epoch=True, sync_dist=True)
        return {"loss": loss.detach()}

    @torch.no_grad()
    def update_memory(self, datamodule, device: torch.device):
        """Use herding to manage exemplars; GEM relies on episodic memory."""
        per_class = self.buffer.exemplars_per_class(num_classes_seen=len(self.seen_classes))
        self.buffer.reduce_exemplars(per_class)
        for c in self.current_classes:
            loader = datamodule.build_class_loader(c, train=True, batch_size=128, shuffle=False, num_workers=datamodule.num_workers)
            self.buffer.build_exemplars_herding(self, self.feature_extractor, loader, c, per_class, device)
