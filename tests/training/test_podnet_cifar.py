import os.path as osp

import lightning as L
import pytest
from lightning.pytorch.loggers import WandbLogger

import wandb
from lycil.constants import _EXP_NAME
from lycil.data.hfmodule import HFDataModule
from lycil.learner.podnet import PODNet

BUFFER_SIZE_PER_CLASS = 20


@pytest.mark.runs_on(["cuda"])
def test_podnet_cifar100(is_dummy_training: bool):
    if is_dummy_training:
        DATAPATH = "data/cifar10"
        N_CLASS_PER_TASK = [1, 1]
        LABEL_COL = "label"
        EPOCHS_PER_TASK = 1
        EPOCHS_PER_TASK_MEMORY = 1
        USE_PRETRAIN_WEIGHTS = False
    else:
        DATAPATH = "data/cifar100"
        N_CLASS_PER_TASK = [20, 20, 20, 20, 20]
        LABEL_COL = "fine_label"
        EPOCHS_PER_TASK = 80
        EPOCHS_PER_TASK_MEMORY = 20
        USE_PRETRAIN_WEIGHTS = True
    if not osp.exists(DATAPATH):
        pytest.skip("Data path does not exist.")
        return

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
        buffer_kwargs={"mem_size_per_class": BUFFER_SIZE_PER_CLASS},
    )
    model = PODNet(
        backbone_args={
            "name": "resnet50",
            "pretrained": USE_PRETRAIN_WEIGHTS,
            "cifar": True,
        },
        head="cosine",
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
        distill_lambda=0.1,
    )

    for task_idx, _ in enumerate(N_CLASS_PER_TASK):
        dm.set_current_task(task_idx)

        # use training data, without buffer
        dm.use_buffer = False
        dm.train_filter_fn = None
        trainer = L.Trainer(
            max_epochs=EPOCHS_PER_TASK,
            sync_batchnorm=True,
            enable_checkpointing=False,
            enable_progress_bar=False,
            precision="16-mixed",
            logger=WandbLogger(
                name=f"podnet_cifar100_{'pretrained_' if USE_PRETRAIN_WEIGHTS else ''}task{task_idx}",
                project="lycil",
                log_model=False,
                tags=["podnet", "cifar100"]
                + ["pretrained" if USE_PRETRAIN_WEIGHTS else "random_init"],
                group=_EXP_NAME,
            ),
            check_val_every_n_epoch=10,
            log_every_n_steps=1000,
        )
        trainer.fit(model, datamodule=dm)

        # use data from buffer only, do not use training data
        dm.use_buffer = True
        dm.train_filter_fn = lambda e: False
        trainer = L.Trainer(
            max_epochs=EPOCHS_PER_TASK_MEMORY,
            sync_batchnorm=True,
            enable_checkpointing=False,
            enable_progress_bar=False,
            precision="16-mixed",
            logger=WandbLogger(
                name=f"podnet_cifar100_{'pretrained_' if USE_PRETRAIN_WEIGHTS else ''}task{task_idx}_memory",
                project="lycil",
                log_model=False,
                tags=["podnet", "cifar100"]
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
    test_podnet_cifar100(is_dummy_training=False)
