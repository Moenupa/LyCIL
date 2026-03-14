import os.path as osp

import lightning as L
import pytest
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import wandb
from lycil.backbone import ConvNetArgs
from lycil.constants import EXP_NAME
from lycil.data.hfmodule import HFDataModule
from lycil.learner.ewc import EWC
from lycil.test_constants import (
    CIFAR10_LABEL_COL,
    CIFAR10_PATH,
    CIFAR100_LABEL_COL,
    CIFAR100_PATH,
)
from tests.training.log_utils import log_statistics_to_wandb


@pytest.mark.slow
@pytest.mark.runs_on(["cuda"])
@pytest.mark.xdist_group("training")
def test_ewc_cifar100(device: str, is_dummy_training: bool):
    if is_dummy_training:
        DATAPATH, LABEL_COL = CIFAR10_PATH, CIFAR10_LABEL_COL
        N_CLASS_PER_TASK = [1, 1]
        EPOCHS_PER_TASK = 1
        USE_PRETRAIN_WEIGHTS = True
    else:
        DATAPATH, LABEL_COL = CIFAR100_PATH, CIFAR100_LABEL_COL
        N_CLASS_PER_TASK = [20, 20, 20, 20, 20]
        EPOCHS_PER_TASK = 160
        USE_PRETRAIN_WEIGHTS = False
    if not osp.exists(DATAPATH):
        pytest.skip("Data path does not exist.")
        return

    L.seed_everything(42)
    dm = HFDataModule(
        DATAPATH,
        transform_name=osp.basename(DATAPATH),
        num_classes_per_task=N_CLASS_PER_TASK,
        label_column_name=LABEL_COL,  # 100 classes
        split_map={"train": "test", "val": "test"}
        if is_dummy_training
        else {"val": "test"},
    )
    model = EWC(
        backbone_args=ConvNetArgs(
            name="resnet50", pretrained=USE_PRETRAIN_WEIGHTS, cifar=True
        ),
        head="linear",
        per_task_optim_args={
            # for all tasks, use the same optimizer kwargs
            "default": {
                "type": "sgd",
                "lr": 0.1,
                "momentum": 0 if USE_PRETRAIN_WEIGHTS else 0.9,
                "weight_decay": 5e-4,
            },
        },
        per_task_sched_args={
            # for all tasks, use the same scheduler kwargs
            "default": {
                "type": "linear_warmup_cosine_annealing",
                "warmup_epochs": 0 if EPOCHS_PER_TASK == 1 else 10,
                "max_epochs": EPOCHS_PER_TASK,
            },
        },
        lambda_ewc=1000,
        fisher_max=0.0001,
    )
    statistics_summary = {}
    for task_idx, _ in enumerate(N_CLASS_PER_TASK):
        dm.set_current_task(task_idx)

        trainer = L.Trainer(
            accelerator=device,
            max_epochs=EPOCHS_PER_TASK,
            sync_batchnorm=True,
            enable_checkpointing=False,
            precision="16-mixed",
            logger=WandbLogger(
                name=f"ewc/task{task_idx}",
                tags=["ewc", "cifar100", str(N_CLASS_PER_TASK)],
                group=f"ewc/{EXP_NAME}",
                offline=is_dummy_training,
            ),
            check_val_every_n_epoch=1,
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
            gradient_clip_val=1.0,
        )
        trainer.fit(model, datamodule=dm)
        test_outputs = trainer.test(
            model=model,
            datamodule=dm,
            verbose=False,
            ckpt_path=None,
        )
        statistics_summary[task_idx] = test_outputs
        log_statistics_to_wandb(trainer, statistics_summary)

        wandb.finish()


if __name__ == "__main__":
    test_ewc_cifar100(device="cuda", is_dummy_training=False)
