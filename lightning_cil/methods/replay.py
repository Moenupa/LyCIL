import torch
import torch.nn.functional as F
from .base import BaseIncremental

class Replay(BaseIncremental):
    """Exemplar replay baseline: CE on mixed stream (current + exemplars)."""
    def training_step(self, batch, batch_idx):
        x, y = batch  # datamodule mixes exemplars already
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def update_memory(self, datamodule, device: torch.device):
        per_class = self.buffer.exemplars_per_class(num_classes_seen=len(self.seen_classes))
        self.buffer.reduce_exemplars(per_class)
        for c in self.current_classes:
            loader = datamodule.build_class_loader(c, train=True, batch_size=128, shuffle=False, num_workers=datamodule.num_workers)
            self.buffer.build_exemplars_herding(self, self.feature_extractor, loader, c, per_class, device)
