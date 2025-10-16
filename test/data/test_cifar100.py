import pytest

from lightning_cil.data.modules.cifar100 import CIFAR100DataModule


@pytest.mark.parametrize("num_class_per_task", [1, 2, 10])
def test_cifar100_datamodule(num_class_per_task: int):
    dm = CIFAR100DataModule(
        root="/public/datasets/cifar",
        num_class_per_task=num_class_per_task,
    )
    dm.prepare_data()
    dm.setup()

    assert dm.num_class_total == 100
    assert dm.num_class_per_task == num_class_per_task
    assert dm.num_tasks == 100 // num_class_per_task


@pytest.mark.parametrize("task_id", [0, 5, 9])
def test_cifar100_datamodule_task(task_id: int):
    dm = CIFAR100DataModule(
        root="/public/datasets/cifar",
        num_class_per_task=10,
    )
    dm.prepare_data()
    dm.setup()

    dm.set_task(task_id)

    assert len(dm.classes_current) == 10
    assert set(dm.classes_current) == set(range(task_id * 10, (task_id + 1) * 10))
    assert len(dm.classes_seen) == (task_id + 1) * 10
    assert set(dm.classes_seen) == set(range(0, (task_id + 1) * 10))

    train_dataset = dm._dataset_train
    test_dataset = dm._dataset_test
    assert len(train_dataset) == 5000
    assert len(test_dataset) == 1000 * (task_id + 1)


@pytest.mark.cuda
def test_cifar100_training():
    # a minimal training loop to test cifar100 datamodule with gpu training
    # good if no error raised
    import lightning as pl
    import torch.nn.functional as F
    from lightning import Trainer
    from torch import nn
    from torch.optim import Adam
    from torchmetrics.classification.accuracy import Accuracy
    from torchvision.models import resnet18

    class DummyClassifier(pl.LightningModule):
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

    dm = CIFAR100DataModule(
        root="/public/datasets/cifar",
        num_class_per_task=1,
        batch_size=2,
    )

    # gpu test
    model = DummyClassifier(num_classes=100)
    trainer = Trainer(accelerator="cuda", max_epochs=1)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
