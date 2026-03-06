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


class OffsetWandbLogger(WandbLogger):
    def __init__(self, step_offset: int = 0, epoch_offset: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.step_offset = step_offset
        self.epoch_offset = epoch_offset

    def log_metrics(self, metrics, step=None):
        if step is not None:
            step += self.step_offset

        if "epoch" in metrics and metrics["epoch"] is not None:
            metrics["epoch"] += self.epoch_offset

        return super().log_metrics(metrics, step=step)


def should_use_distill(task_idx: int, use_buffer: bool) -> bool:
    # 典型逻辑：非首任务、且不是 memory/buffer 阶段才 distill
    return (task_idx > 0) and (not use_buffer)


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
        EPOCHS_PER_TASK = 20
        EPOCHS_PER_TASK_MEMORY = 10
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
        split_map= {"train": "train", "val": "test", "test": "test"},
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
            # for all tasks, use the same optimizer kwargs
            "default": {
                "type": "sgd",
                "lr": 0.1,
                "weight_decay": 5e-4,
            },
            "buffer": {
                "type": "sgd",
                "lr": 0.005,
                # "lr": 0.,
                "weight_decay": 5e-4,
            },
        },
        per_task_sched_args={
            # for all tasks, use the same scheduler kwargs
            # "default": {
            #     "type": "linear_warmup_cosine_annealing",
            #     "warmup_epochs": 0 if EPOCHS_PER_TASK == 1 else 10,
            #     "max_epochs": EPOCHS_PER_TASK,
            # },
            "default": {
                "type": "cosine_annealing",
                "T_max": EPOCHS_PER_TASK,
            },
            "buffer": {
                "type":None # No scheduler during buffer training
                # "type": "linear_warmup_cosine_annealing",
                # "warmup_epochs": 5,
                # "max_epochs": EPOCHS_PER_TASK_MEMORY,
                # "type": "cosine_annealing",
                # "T_max": EPOCHS_PER_TASK_MEMORY,
            },
        },
        lambda_spatial=5.0,
        lambda_flat=1.0,
    )

    for task_idx, _ in enumerate(N_CLASS_PER_TASK):
        model.using_distill = should_use_distill(task_idx, use_buffer=False)
        model.buffer_training = False  # Not buffer stage
        dm.set_current_task(task_idx)
        # use training data, without buffer
        dm.use_buffer = True
        dm.train_filter_fn = None
        logger1 = OffsetWandbLogger(
            resume="allow",
            name=f"force_reset_unfixed_b_mask_distill_b_wo_warmup_podnet_cifar100_{'pretrained_' if USE_PRETRAIN_WEIGHTS else ''}task{task_idx}",
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
            # log_every_n_steps=10,
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
        )
        trainer1.fit(model, datamodule=dm)

        if task_idx > 0:
            model.using_distill = should_use_distill(task_idx, use_buffer=False)
            model.buffer_training = True  # Not buffer stage
            # use data from buffer only, do not use training data
            if hasattr(model.classifier, "old_head") and model.classifier.old_head is not None:
                model.classifier.old_head.requires_grad_(False)
            # model.backbone.eval()
            dm.use_buffer = True
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
                # log_every_n_steps=10,
                callbacks=[LearningRateMonitor(logging_interval="epoch")],
            )
            trainer2.fit(model, datamodule=dm)

        # trainer.validate(model, datamodule=dm)
        wandb.finish()


if __name__ == "__main__":
    test_podnet_cifar100(is_dummy_training=False)
