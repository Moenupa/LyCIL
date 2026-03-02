import os.path as osp

import lightning as L
import pytest
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import wandb
from lycil.constants import _EXP_NAME
from lycil.data.hfmodule import HFDataModule
from lycil.learner.podnet import PODNet

BUFFER_SIZE_PER_CLASS = 20


@pytest.mark.slow
@pytest.mark.runs_on(["cuda"])
def test_podnet_cifar100(is_dummy_training: bool):
    if is_dummy_training:
        DATAPATH = "/ppio_net0/datasets/cifar10"
        N_CLASS_PER_TASK = [1, 1]
        LABEL_COL = "label"
        EPOCHS_PER_TASK = 1
        EPOCHS_PER_TASK_MEMORY = 1
        USE_PRETRAIN_WEIGHTS = False
    else:
        DATAPATH = "/ppio_net0/datasets/cifar100"
        N_CLASS_PER_TASK = [20, 20, 20, 20, 20]
        LABEL_COL = "fine_label"
        EPOCHS_PER_TASK = 160
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
        train_loader_kwargs={"batch_size": 128, "shuffle": True, "num_workers": 10},
        val_loader_kwargs={"batch_size": 128, "shuffle": False, "num_workers": 10},
        test_loader_kwargs={"batch_size": 128, "shuffle": False, "num_workers": 10},
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
            # for buffer training, small learning rate
            -2: {
                "type": "sgd",
                "lr": 0.005,
                "weight_decay": 5e-4,
            },
            # for all tasks, use the same optimizer kwargs
            -1: {
                "type": "sgd",
                "lr": 0.1,
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
        lambda_spatial=5.0,
        lambda_flat=1.0,
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
            enable_progress_bar=True,
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
            log_every_n_steps=10,
        )
        trainer.fit(model, datamodule=dm)

        # use data from buffer only, do not use training data
        dm.use_buffer = True
        dm.train_filter_fn = lambda e: False
        # to bypass head expansion, see `BaseLearner.sync_with_datamodule()`
        # and get special training optimizer kwargs with key -2
        model.set_task_id(-2)
        trainer = L.Trainer(
            max_epochs=EPOCHS_PER_TASK_MEMORY,
            sync_batchnorm=True,
            enable_checkpointing=False,
            enable_progress_bar=True,
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
            # log_every_n_steps=10,
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
        )
        trainer.fit(model, datamodule=dm)
        # reset after memory training
        model.set_task_id(task_idx)
        trainer.validate(model, datamodule=dm)

        wandb.finish()



if __name__ == "__main__":
    test_podnet_cifar100(is_dummy_training=False)
