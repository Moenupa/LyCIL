from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseIncremental

class LUCIR(BaseIncremental):
    """LUCIR with cosine classifier, less-forget and margin ranking constraints.

    References:
    - Hou et al. Learning a Unified Classifier Incrementally via Rebalancing (CVPR 2019).
    """
    def __init__(
        self,
        *args,
        lucir_margin: float = 0.5,
        lucir_distill_T: float = 2.0,
        lucir_K: int = 2,
        lambda_less_forget: float = 5.0,
        lambda_margin: float = 1.0,
        lambda_distill: float = 1.0,
        **kwargs
    ):
        kwargs.setdefault("head", "cosine")  # enforce cosine by default
        super().__init__(*args, **kwargs)
        self.lucir_margin = float(lucir_margin)
        self.lucir_distill_T = float(lucir_distill_T)
        self.lucir_K = int(lucir_K)
        self.lambda_less_forget = float(lambda_less_forget)
        self.lambda_margin = float(lambda_margin)
        self.lambda_distill = float(lambda_distill)

    def training_step(self, batch, batch_idx):
        x, y = batch
        f = self.feature_extractor(x)  # (B,d)
        f_norm = F.normalize(f, dim=1)
        logits = self.classifier(f)    # cosine head inside

        # Cross-entropy on all seen classes
        loss_ce = F.cross_entropy(logits, y)

        # Less-forget constraint: encourage feature consistency vs prev model
        loss_lf = torch.tensor(0.0, device=x.device)
        if self.prev_model is not None:
            with torch.no_grad():
                f_prev = self.prev_model.feature_extractor(x)
                f_prev = F.normalize(f_prev, dim=1)
            # cosine alignment
            cos_sim = (f_norm * f_prev).sum(dim=1)
            loss_lf = 1 - cos_sim.mean()

        # Margin ranking loss to push down old-class logits
        loss_margin = torch.tensor(0.0, device=x.device)
        if len(self.seen_classes) > 0:
            old_idx = torch.tensor(sorted(self.seen_classes), device=x.device, dtype=torch.long)
            # clamp indices within current head size
            old_idx = old_idx[old_idx < logits.size(1)]
            if old_idx.numel() > 0:
                gt = logits.gather(1, y.view(-1,1)).squeeze(1)  # (B,)
                old_scores = logits.index_select(1, old_idx)     # (B,|old|)
                # take top-K old scores
                topk_old = old_scores.topk(min(self.lucir_K, old_scores.size(1)), dim=1).values  # (B,K)
                # ranking: gt >= topk_old + margin
                margin = self.lucir_margin
                loss_margin = F.relu(topk_old + margin - gt.unsqueeze(1)).mean()

        # Distillation with temperature T on old classes
        loss_distill = torch.tensor(0.0, device=x.device)
        if self.prev_model is not None and len(self.seen_classes) > 0:
            with torch.no_grad():
                prev_logits = self.prev_model.classifier(self.prev_model.feature_extractor(x))
            old_idx = torch.tensor(sorted(self.seen_classes), device=x.device, dtype=torch.long)
            old_idx = old_idx[old_idx < logits.size(1)]
            if old_idx.numel() > 0:
                T = self.lucir_distill_T
                p = F.log_softmax(logits.index_select(1, old_idx) / T, dim=1)
                q = F.softmax(prev_logits.index_select(1, old_idx) / T, dim=1)
                loss_distill = F.kl_div(p, q, reduction="batchmean") * (T * T)

        loss = loss_ce + self.lambda_less_forget * loss_lf + self.lambda_margin * loss_margin + self.lambda_distill * loss_distill
        self.log_dict(
            {
                "train/loss": loss,
                "train/ce": loss_ce,
                "train/lf": loss_lf,
                "train/margin": loss_margin,
                "train/distill": loss_distill,
            },
            prog_bar=True, on_epoch=True, sync_dist=True
        )
        return loss

    @torch.no_grad()
    def update_memory(self, datamodule, device: torch.device):
        """Herding exemplars and rebalance per class quota."""
        per_class = self.buffer.exemplars_per_class(num_classes_seen=len(self.seen_classes))
        self.buffer.reduce_exemplars(per_class)
        for c in self.current_classes:
            loader = datamodule.build_class_loader(c, train=True, batch_size=128, shuffle=False, num_workers=datamodule.num_workers)
            self.buffer.build_exemplars_herding(self, self.feature_extractor, loader, c, per_class, device)
