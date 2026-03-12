import os.path as osp
import lightning as L
import pytest
from lightning.pytorch.callbacks import LearningRateMonitor
import wandb
from lycil.constants import _EXP_NAME
from lycil.data.hfmodule import HFDataModule
from lycil.learner.podnet import PODNet
from lycil.backbone import ConvNetArgs
from tests.training.log_utils import log_statistics_to_wandb,OffsetWandbLogger




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
        BUFFER_SIZE_PER_CLASS = 20
    else:
        DATAPATH = "/ppio_net0/datasets/cifar100"
        N_CLASS_PER_TASK = [20] * 5
        LABEL_COL = "fine_label"
        EPOCHS_PER_TASK = 160
        EPOCHS_PER_TASK_MEMORY = 20
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
        val_loader_kwargs={"batch_size": 128, "shuffle": False, "num_workers": 8},
        test_loader_kwargs={"batch_size": 128, "shuffle": False, "num_workers": 8},
        split_map={"train": "train", "val": "test", "test": "test"},
        # Use an adaptive total-memory budget so early tasks can temporarily
        # occupy the slots of unseen future classes.
        buffer_kwargs={"mem_size": total_buffer_size},

    )
    model = PODNet(
        backbone_args=ConvNetArgs(name="resnet50", pretrained=USE_PRETRAIN_WEIGHTS, cifar=True),
        head="cosine",
        per_task_optim_args={
            # for buffer training, small learning rate
            # for all tasks, use the same optimizer kwargs
            "default": {
                "type": "sgd",
                "lr": 0.1,
                "momentum": 0 if USE_PRETRAIN_WEIGHTS else 0.9,
                "weight_decay": 5e-4,
            },
            "buffer": {
                "type": "sgd",
                "lr": 0.005,
                "momentum": 0 if USE_PRETRAIN_WEIGHTS else 0.9,
                "weight_decay": 5e-4,
            },
        },
        per_task_sched_args={
            # "default": {
            #     "type": "cosine_annealing",
            #     "T_max": EPOCHS_PER_TASK,
            # },
            # "buffer": {
            #     "type": "cosine_annealing",
            #     "T_max": EPOCHS_PER_TASK_MEMORY,
            # },
            "default": {
                "type": "linear_warmup_cosine_annealing",
                "warmup_epochs": 0 if EPOCHS_PER_TASK == 1 else 10,
                "max_epochs": EPOCHS_PER_TASK,
            },
            "buffer": {
                "type": "linear_warmup_cosine_annealing",
                "warmup_epochs": 0 if EPOCHS_PER_TASK == 1 else 10,
                "max_epochs": EPOCHS_PER_TASK_MEMORY,
            },

        },
        lambda_spatial=5.0,
        lambda_flat=1.0,
        buffer_args={
            "selection": "herding",
            "seed": 42,
            "loader_kwargs": {
                "batch_size": 128,
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
        }
    )

    statistics_summary = {}

    for task_idx, _ in enumerate(N_CLASS_PER_TASK):
        model.train()
        model.using_distill = task_idx > 0
        model.need_snapshot_old = task_idx == 0
        model.buffer_training = False
        dm.set_current_task(task_idx)
        dm.train_filter_fn = None

        logger1 = OffsetWandbLogger(
            resume="allow",
            name=f"podnet_cifar100_{'pretrained_' if USE_PRETRAIN_WEIGHTS else ''}T{len(N_CLASS_PER_TASK)}_task{task_idx}",
            project="lycil",
            log_model=False,
            tags=["podnet", "cifar100"] + ["pretrained" if USE_PRETRAIN_WEIGHTS else "random_init"],
            group=_EXP_NAME,
        )
        trainer1 = L.Trainer(
            max_epochs=EPOCHS_PER_TASK,
            sync_batchnorm=True,
            enable_checkpointing=False,
            enable_progress_bar=True,
            precision="16-mixed",
            logger=logger1,
            check_val_every_n_epoch=1,
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
        )
        trainer1.fit(model, datamodule=dm)
        final_trainer = trainer1

        if task_idx > 0:
            model.using_distill = True
            model.buffer_training = True
            model.need_snapshot_old = True

            # model.backbone.eval()
            # model.backbone.requires_grad_(False)
            dm.train_filter_fn = lambda e: False

            logger2 = OffsetWandbLogger(
                resume="allow",
                step_offset=trainer1.global_step,  # 关键：把第二段的 step 往后平移
                epoch_offset=trainer1.current_epoch + 1,  # epoch 从 0 开始；+1 才能连续
                name=logger1._name,  # 可选：同名
                project="lycil",
                log_model=False,
                tags=["podnet", "cifar100"] + ["pretrained" if USE_PRETRAIN_WEIGHTS else "random_init"],
                group=_EXP_NAME,
            )

            trainer2 = L.Trainer(
                max_epochs=EPOCHS_PER_TASK_MEMORY,
                sync_batchnorm=True,
                enable_checkpointing=False,
                enable_progress_bar=True,
                precision="16-mixed",
                logger=logger2,
                check_val_every_n_epoch=1,
                callbacks=[LearningRateMonitor(logging_interval="epoch")],
            )
            trainer2.fit(model, datamodule=dm)
            final_trainer = trainer2
            # model.backbone.requires_grad_(True)

        test_outputs = final_trainer.test(
            model=model,
            datamodule=dm,
            verbose=False,
            ckpt_path=None,
        )
        statistics_summary[task_idx] = test_outputs
        log_statistics_to_wandb(final_trainer, statistics_summary)

        wandb.finish()


if __name__ == "__main__":
    test_podnet_cifar100(is_dummy_training=False)
