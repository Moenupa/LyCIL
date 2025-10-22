# a minimal test to check cifar100 datamodule works with a training loop

import lightning as L
import pytest
from torch import cuda, nn
from torch.nn import functional as F
from torch.optim import Adam
from torchmetrics.classification.accuracy import Accuracy
from torchvision.models import resnet18

from lycil.data.modules.cifar100 import CIFAR100DataModule


def _is_cuda_available() -> bool:
    return cuda.is_available()


def _is_npu_available() -> bool:
    try:
        import torch_npu

        return torch_npu.npu.is_available()
    except Exception:
        return False


CHECKER = {
    "cuda": _is_cuda_available,
    "npu": _is_npu_available,
}


class DummyClassifier(L.LightningModule):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.acc = Accuracy(num_classes=num_classes, task="multiclass", top_k=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.acc(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.acc(logits, y)
        self.log("test/loss", loss)
        self.log("test/acc", acc)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        return optimizer


@pytest.mark.parametrize("accelerator", ["cuda", "npu"])
def test_cifar100_training(accelerator: str):
    if not CHECKER[accelerator]():
        pytest.skip(f"{accelerator} is not available")

    dm = CIFAR100DataModule(
        root="data/cifar",
        num_class_per_task=1,
        batch_size=32,
    )

    model = DummyClassifier(num_classes=100)
    trainer = L.Trainer(accelerator=accelerator, max_epochs=1)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
