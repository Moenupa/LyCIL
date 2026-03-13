import os.path as osp

import lightning as L
import pytest
from lightning.pytorch.loggers import WandbLogger


import wandb
from lycil.constants import _EXP_NAME
from lycil.data.hfmodule import HFDataModule
from lycil.learner.icarl import ICaRL

from tests.training.constants import (
    CIFAR10_LABEL_COL,
    CIFAR10_PATH,
    CIFAR100_LABEL_COL,
    CIFAR100_PATH,
    CONVNET_ARGS,
    TEST_LOADER_KWARGS,
    VAL_LOADER_KWARGS,
)
from lightning.pytorch.callbacks import LearningRateMonitor
from lycil.backbone import ConvNetArgs

from tests.training.log_utils import log_statistics_to_wandb

@pytest.mark.slow
@pytest.mark.runs_on(["cuda"])
@pytest.mark.xdist_group("training")
def test_icarl_cifar100(device: str, is_dummy_training: bool):
    if is_dummy_training:
        DATAPATH, LABEL_COL = CIFAR10_PATH, CIFAR10_LABEL_COL
        N_CLASS_PER_TASK = [1, 1]
        EPOCHS_PER_TASK = 1
    else:
        DATAPATH = "/ppio_net0/datasets/cifar100"
        N_CLASS_PER_TASK = [20] * 5
        LABEL_COL = "fine_label"
        EPOCHS_PER_TASK = 160
        USE_PRETRAIN_WEIGHTS = False
        BUFFER_SIZE_PER_CLASS = 20
    if not osp.exists(DATAPATH):
        pytest.skip("Data path does not exist.")
        return

    L.seed_everything(42)
    total_buffer_size = BUFFER_SIZE_PER_CLASS * sum(N_CLASS_PER_TASK)
    dm = HFDataModule(
        DATAPATH,
        transform_name=osp.basename(DATAPATH),
        num_classes_per_task=N_CLASS_PER_TASK,
        label_column_name=LABEL_COL,  # 100 classes
        train_loader_kwargs={"batch_size": 128, "shuffle": True, "num_workers": 8},
        val_loader_kwargs=VAL_LOADER_KWARGS,
        test_loader_kwargs=TEST_LOADER_KWARGS,
        split_map={"train": "train", "val": "test", "test": "test"},
        # buffer_kwargs={"mem_size_per_class": BUFFER_SIZE_PER_CLASS},
        # Use an adaptive total-memory budget so early tasks can temporarily
        # occupy the slots of unseen future classes.
        buffer_kwargs={"mem_size": total_buffer_size},
        # buffer_kwargs={"mem_size_per_class": BUFFER_SIZE_PER_CLASS},
    )
    model = ICaRL(
        backbone_args=ConvNetArgs(name="resnet50", pretrained=USE_PRETRAIN_WEIGHTS, cifar=True),
        head="linear",
        per_task_optim_args={
            # for all tasks, use the same optimizer kwargs
            "default": {
                "type": "sgd",
                "lr": 0.08,
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
            # "default": {
            #     "type": "multi_step_lr",
            #     "milestones": [60, 120, 170],
            #     "gamma": 0.1,
            # },

        },
        distill_T=1.0,
        distill_lambda=0.1,
        buffer_args={
            "selection": "herding",
            "seed": 42,
            "loader_kwargs": {
                "batch_size": 256,
                "shuffle": False,
                "num_workers": 8,
            },
            "nme_eval": {
                "enable": True,
                "topk": 1,
                "dynamic_old": True,
                "dynamic_new": True,
                "every_n_epochs": EPOCHS_PER_TASK,
            },
        },

    )

    statistics_summary={}
    for task_idx, _ in enumerate(N_CLASS_PER_TASK):
        dm.set_current_task(task_idx)

        trainer = L.Trainer(
            accelerator=device,
            max_epochs=EPOCHS_PER_TASK,
            sync_batchnorm=True,
            enable_checkpointing=False,
            enable_progress_bar=True,
            precision="16-mixed",
            logger=WandbLogger(
                name=f"icarl_cifar100_T{len(N_CLASS_PER_TASK)}_task{task_idx}",
                project="lycil",
                log_model=False,
                tags=["icarl", "cifar100"],
                group=_EXP_NAME,
            ),
            check_val_every_n_epoch=1,
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
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
        break


if __name__ == "__main__":
    test_icarl_cifar100(device="cuda", is_dummy_training=False)
