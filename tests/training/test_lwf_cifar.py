import os
import os.path as osp

os.environ["X_COLUMN_NAME"] = "img"

import lightning as L
import pytest
from lightning.pytorch.loggers import WandbLogger

import wandb
from lycil.constants import _EXP_NAME
from lycil.data.hfmodule import HFDataModule
from lycil.learner.lwf import LWF

DUMMY_TRAINING = os.environ.get("DUMMY", "1") == "1"
CUDA_AVAILABLE = len(os.environ.get("CUDA_VISIBLE_DEVICES", "")) > 0


@pytest.fixture
def cifar_args() -> list | None:
    args = (
        ["data/cifar10", [1, 1], "label", 1, False]
        if DUMMY_TRAINING
        else ["data/cifar100", [20, 20, 20, 20, 20], "fine_label", 80, True]
    )
    if osp.exists(args[0]):  # ty: ignore[invalid-argument-type]
        return args

    return None


def test_lwf_cifar100(cifar_args: list | None):
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available.")
        return
    if cifar_args is None:
        pytest.skip("Data path does not exist.")
        return

    DATAPATH, N_CLASS_PER_TASK, LABEL_COL, EPOCHS_PER_TASK, USE_PRETRAIN_WEIGHTS = (
        cifar_args
    )

    L.seed_everything(42)
    dm = HFDataModule(
        DATAPATH,
        transform_name=osp.basename(DATAPATH),
        num_classes_per_task=N_CLASS_PER_TASK,
        label_column_name=LABEL_COL,  # 100 classes
        train_loader_kwargs={"batch_size": 64, "shuffle": True, "num_workers": 10},
        val_loader_kwargs={"shuffle": False, "num_workers": 10},
        test_loader_kwargs={"shuffle": False, "num_workers": 10},
        split_map={"train": "test", "val": "test"}
        if EPOCHS_PER_TASK == 1
        else {"val": "test"},
    )
    model = LWF(
        backbone_args={
            "name": "resnet50",
            "pretrained": USE_PRETRAIN_WEIGHTS,
            "cifar": True,
        },
        head="linear",
        per_task_optim_args={
            # for all tasks, use the same optimizer kwargs
            -1: {
                "type": "sgd",
                "lr": 0.3,
                "weight_decay": 5e-4,
            },
        },
        per_task_sched_args={
            # for all tasks, use the same scheduler kwargs
            -1: {
                "type": "linear_warmup_cosine_annealing",
                "warmup_epochs": 0 if EPOCHS_PER_TASK == 1 else 10,
                "max_epochs": EPOCHS_PER_TASK,
            }
        },
        distill_T=1.0,
        distill_lambda=0.1,
    )

    for task_idx, _ in enumerate(N_CLASS_PER_TASK):
        dm.set_current_task(task_idx)

        trainer = L.Trainer(
            max_epochs=EPOCHS_PER_TASK,
            sync_batchnorm=True,
            enable_checkpointing=False,
            enable_progress_bar=False,
            precision="16-mixed",
            logger=WandbLogger(
                name=f"lwf_cifar100_{'pretrained_' if USE_PRETRAIN_WEIGHTS else ''}task{task_idx}",
                project="lycil",
                log_model=False,
                tags=["lwf", "cifar100"]
                + ["pretrained" if USE_PRETRAIN_WEIGHTS else "random_init"],
                group=_EXP_NAME,
            ),
            check_val_every_n_epoch=10,
            log_every_n_steps=1000,
        )
        trainer.fit(model, datamodule=dm)
        trainer.validate(model, datamodule=dm)

        wandb.finish()


if __name__ == "__main__":
    pytest.main()
