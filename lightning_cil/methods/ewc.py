from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseIncremental

class EWC(BaseIncremental):
    """Elastic Weight Consolidation (Kirkpatrick et al., PNAS 2017)."""
    def __init__(
        self, *args,
        ewc_lambda: float = 30.0,
        ewc_samples_per_class: int = 200,
        **kwargs
    ):
        kwargs.setdefault("mem_size", 0)  # no exemplars needed
        super().__init__(*args, **kwargs)
        self.ewc_lambda = float(ewc_lambda)
        self.ewc_samples_per_class = int(ewc_samples_per_class)
        self._fisher: Dict[str, torch.Tensor] = {}
        self._theta_star: Dict[str, torch.Tensor] = {}

    def ewc_penalty(self) -> torch.Tensor:
        if not self._fisher:
            return torch.tensor(0.0, device=self.device)
        penalty = 0.0
        for n, p in self.named_parameters():
            if n in self._fisher:
                penalty = penalty + (self._fisher[n] * (p - self._theta_star[n])**2).sum()
        return penalty

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        loss = loss + self.ewc_lambda * self.ewc_penalty()
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def consolidate(self, datamodule, device: torch.device):
        """Estimate diagonal Fisher on current task and store theta*."""
        # snapshot theta*
        self._theta_star = {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad}

        # Init fisher diag
        fisher = {n: torch.zeros_like(p, device=device) for n, p in self.named_parameters() if p.requires_grad}
        self.eval()
        # data loader: sample subset from current task train data
        loader = datamodule.train_dataloader()
        num_seen = 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            self.zero_grad(set_to_none=True)
            logits = self(x)
            # Use log prob of predicted class for FIM approximation
            logp = F.log_softmax(logits, dim=1)
            probs = logp.exp().detach()
            # take the label y (or prediction); here use y
            loss = F.nll_loss(logp, y, reduction='mean')
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None and p.requires_grad:
                    fisher[n] += (p.grad.detach() ** 2)
            num_seen += x.size(0)
            if num_seen >= self.ewc_samples_per_class * max(1, len(datamodule.current_classes)):
                break

        # average
        for n in fisher:
            fisher[n] /= float(max(1, num_seen))
        self._fisher = fisher

    def update_memory(self, *args, **kwargs):
        # EWC uses no exemplars by default
        return
