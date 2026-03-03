from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseLearner

if TYPE_CHECKING:
    from ..data.hfmodule import HFDataModule


def harmonize_keyname(
    key: str,
    inverse: bool = False,
    _from: str = ".",
    _to: str = "-",
) -> str:
    # pytorch restricts the use of "." in parameter names
    # we replace it with "-" when storing in the fisher_dict and mean_dict
    if inverse:
        return key.replace(_to, _from)
    return key.replace(_from, _to)


class EWC(BaseLearner):
    r"""`Elastic Weight Consolidation`_ (Kirkpatrick et al., PNAS 2017).

    Args:
        lambda_ewc (float, optional): Weight for EWC penalty. (default: 1000.0).
        fisher_max (float, optional): Maximum value for Fisher information. (default: 1e-4).
        args: See :class:`BaseLearner` for other args.
        kwargs: See :class:`BaseLearner` for other args.

    .. _Elastic Weight Consolidation:
        https://doi.org/10.1073/pnas.1611835114
    """

    def __init__(
        self, *args, lambda_ewc: float = 1000.0, fisher_max: float = 1e-4, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda_ewc = float(lambda_ewc)
        self.fisher_max = float(fisher_max)

        self.fisher_dict = nn.ParameterDict()
        self.mean_dict = nn.ParameterDict()

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = self.unpack_batch(batch)
        logits: torch.Tensor = self(x)

        if self.task_id > 0:
            # ce on current classes
            loss_ce = F.cross_entropy(
                logits[:, self.num_old_classes :], y - self.num_old_classes
            )
            loss_ewc = self.compute_ewc_loss()
            loss = loss_ce + self.lambda_ewc * loss_ewc
        else:
            # first task, no ewc
            loss_ce = F.cross_entropy(logits, y)
            loss_ewc = None
            loss = loss_ce

        self.log_dict(
            {
                "train/loss": loss,
                "train/ce": loss_ce,
                "train/ewc": loss_ewc or 0.0,
            },
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def harmonize_named_parameters(self, *args, **kwargs):
        for n, p in self.named_parameters(*args, **kwargs):
            # prevent recursive inclusion of fisher_dict and mean_dict
            if "fisher_dict." in n or "mean_dict." in n:
                continue

            yield harmonize_keyname(n), p

    def compute_ewc_loss(self) -> torch.Tensor:
        loss_ewc = torch.tensor(0.0, device=self.device)
        for n, p in self.harmonize_named_parameters():
            # Only consider parameters that were present in the previous task
            if n not in self.fisher_dict:
                continue

            old_p = self.mean_dict[n]
            f = self.fisher_dict[n]
            _loss = torch.sum(f * (p[: len(old_p)] - old_p).pow(2)) / 2

            loss_ewc = loss_ewc + _loss
        return loss_ewc

    def on_train_end(self) -> None:
        dm = self.trainer.datamodule  # ty: ignore[unresolved-attribute]
        self.update_fisher_and_mean(dm)

    @torch.no_grad()
    def update_fisher_and_mean(self, dm: "HFDataModule") -> None:
        new_fisher = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.harmonize_named_parameters()
            if p.requires_grad
        }

        self.train()
        with torch.enable_grad():
            train_loader = dm.train_dataloader()
            for batch in train_loader:
                self.zero_grad()
                x, y = self.unpack_batch(batch, self.device)
                logits = self(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()

                for n, p in self.harmonize_named_parameters():
                    if p.grad is not None:
                        new_fisher[n] += p.grad.pow(2).detach()
        self.zero_grad()

        # inplace normalization + clipping
        for n, f in new_fisher.items():
            new_fisher[n] = torch.clamp(f / len(train_loader), max=self.fisher_max)

        # if old fisher, combine using weighted average
        if len(self.fisher_dict) > 0:
            alpha = self.num_old_classes / self.num_seen_classes
            for n, f in new_fisher.items():
                if n not in self.fisher_dict:
                    continue

                old_f = self.fisher_dict[n]
                old_f_len = len(old_f)
                new_fisher[n][:old_f_len] = alpha * old_f + (1 - alpha) * f[:old_f_len]

        self.fisher_dict = nn.ParameterDict(
            {
                n: nn.Parameter(f.detach(), requires_grad=False)
                for n, f in new_fisher.items()
            }
        )
        self.mean_dict = nn.ParameterDict(
            {
                n: nn.Parameter(p.clone().detach(), requires_grad=False)
                for n, p in self.harmonize_named_parameters()
                if p.requires_grad
            }
        )

    @torch.no_grad()
    def update_memory(self, *args, **kwargs):
        """EWC stores no exemplars."""
        return
