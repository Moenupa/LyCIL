# a minimal test to check cifar10 datamodule works with a training loop

import lightning as L
import pytest
from torch import cuda, nn
from torch.nn import functional as F
from torch.optim import Adam
from torchmetrics.classification.accuracy import Accuracy
from torchvision.models import resnet18

from lycil.data.hfmodule import HFDataModule


def _is_cuda_available() -> bool:
    return cuda.is_available()


def _is_npu_available() -> bool:
    try:
        import torch_npu  # type: ignore

        return torch_npu.npu.is_available()
    except Exception:
        return False


CHECKER = {
    "cuda": _is_cuda_available,
    "npu": _is_npu_available,
}


class DummyClassifier(L.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.acc = Accuracy(num_classes=num_classes, task="multiclass", top_k=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["img"]
        y = batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, *args, **kwargs):
        pass

    def test_step(self, batch, batch_idx):
        x = batch["img"]
        y = batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.acc(logits, y)
        self.log_dict({"test/loss": loss, "test/acc": acc}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        return optimizer


@pytest.mark.parametrize("accelerator", ["cuda", "npu"])
def test_cifar10_training(accelerator: str):
    if not CHECKER[accelerator]():
        pytest.skip(f"{accelerator} is not available")

    dm = HFDataModule(
        "data/cifar10",
        split_map={"val": "test"},
        transform_name="cifar10",
        train_loader_kwargs={
            "batch_size": 32,
            "num_workers": 8,
            "pin_memory": True,
        },
        test_loader_kwargs={
            "num_workers": 8,
        },
    )

    model = DummyClassifier(num_classes=10)
    trainer = L.Trainer(
        accelerator=accelerator,
        max_epochs=1,
        enable_checkpointing=False,
        log_every_n_steps=None,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    test_cifar10_training("cuda")
