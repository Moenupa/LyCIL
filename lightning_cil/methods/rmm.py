import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from .base import BaseIncremental

class PolicyNet(nn.Module):
    """Simple scoring network over features to parameterize selection policy."""
    def __init__(self, d: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
    def forward(self, f):
        return self.net(f).squeeze(1)  # logits

class RMM(BaseIncremental):
    """Reinforced Memory Management: policy scores candidates; REINFORCE-style update."""
    def __init__(self, *args, policy_hidden: int = 256, rmm_samples: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = PolicyNet(self.feature_dim, policy_hidden)
        self.rmm_samples = int(rmm_samples)
        self.policy_optim = None  # created in configure_optimizers

    def configure_optimizers(self):
        cfg = super().configure_optimizers()
        # add policy optimizer
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=max(1e-4, self.hparams.lr * 0.1))
        return cfg

    @torch.no_grad()
    def _evaluate_reward(self, x, y):
        """Use negative CE as reward proxy."""
        logits = self(x)
        return (-F.cross_entropy(logits, y)).item()

    @torch.no_grad()
    def _gather_candidates(self, datamodule, class_id: int, device: torch.device):
        loader = datamodule.build_class_loader(class_id, train=True, batch_size=128, shuffle=False, num_workers=datamodule.num_workers)
        xs, ys = [], []
        for xb, yb in loader:
            xs.append(xb)
            ys.append(yb)
        if not xs:
            return None, None
        x = torch.cat(xs, dim=0).to(device)
        y = torch.cat(ys, dim=0).to(device)
        return x, y

    def _policy_select(self, feats, m):
        logits = self.policy(feats.detach())
        probs = torch.softmax(logits, dim=0)
        # sample without replacement proportional to probs
        idx = torch.multinomial(probs, num_samples=min(m, feats.size(0)), replacement=False)
        return idx, logits

    @torch.no_grad()
    def update_memory(self, datamodule, device: torch.device):
        # REINFORCE per class with a small rollout on a subset
        per_class = self.buffer.exemplars_per_class(num_classes_seen=len(self.seen_classes))
        self.buffer.reduce_exemplars(per_class)

        for c in self.current_classes:
            x, y = self._gather_candidates(datamodule, c, device)
            if x is None:
                continue

            # subsample for quick policy update
            if x.size(0) > self.rmm_samples:
                idx = torch.randperm(x.size(0), device=device)[:self.rmm_samples]
                x_s, y_s = x[idx], y[idx]
            else:
                x_s, y_s = x, y

            # compute features
            f = self.feature_extractor(x_s)
            idx_sel, logits = self._policy_select(f, per_class)

            # reward baseline: mean reward of a random selection of same size
            rand_idx = torch.randperm(x_s.size(0), device=device)[:idx_sel.numel()]
            R_sel = self._evaluate_reward(x_s[idx_sel], y_s[idx_sel])
            R_rand = self._evaluate_reward(x_s[rand_idx], y_s[rand_idx])
            advantage = R_sel - R_rand

            # REINFORCE update on policy
            self.policy_optim.zero_grad(set_to_none=True)
            # negative log-likelihood of selected indices
            logp = torch.log_softmax(logits, dim=0)[idx_sel].sum()
            loss_pg = -advantage * logp
            loss_pg.backward()
            self.policy_optim.step()

            # Finally, store top-m by policy score (greedy)
            with torch.no_grad():
                full_f = self.feature_extractor(x)
                scores = self.policy(full_f)
                topk = torch.topk(scores, k=min(per_class, scores.numel())).indices
                images = [x[i].cpu() for i in topk.tolist()]
                self.buffer.storage[c] = images
