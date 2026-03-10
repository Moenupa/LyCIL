import os.path as osp
import lightning as L
import pytest
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import wandb
from lycil.constants import _EXP_NAME
from lycil.data.hfmodule import HFDataModule
from lycil.learner.podnet import PODNet
from lycil.backbone import ConvNetArgs


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


def need_snapshot_old(task_idx: int, use_buffer: bool) -> bool:
    # task 0: 没有 buffer 阶段，主训练结束后直接 snapshot
    if task_idx == 0:
        return not use_buffer
    # task 1+: 只在 buffer 微调阶段结束后 snapshot
    return use_buffer


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
        N_CLASS_PER_TASK = [20, 20, 20, 20, 20]
        LABEL_COL = "fine_label"
        EPOCHS_PER_TASK = 1
        EPOCHS_PER_TASK_MEMORY = 1
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
        train_loader_kwargs={"batch_size": 128, "shuffle": True, "num_workers": 10},
        val_loader_kwargs={"batch_size": 128, "shuffle": False, "num_workers": 10},
        test_loader_kwargs={"batch_size": 128, "shuffle": False, "num_workers": 10},
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
            "default": {
                "type": "cosine_annealing",
                "T_max": EPOCHS_PER_TASK,
            },
            "buffer": {
                "type": "cosine_annealing",
                "T_max": EPOCHS_PER_TASK_MEMORY,
            },
        },
        lambda_spatial=5.0,
        lambda_flat=1.0,
        buffer_args={
            "selection": "random",
            "seed": 42,
            "loader_kwargs": {
                "batch_size": 128,
                "shuffle": False,
                "num_workers": 8,
            },
            "nme_eval": {
                "enable": False,
                "topk": 1,
                "dynamic_old": True,
                "dynamic_new": True,
                "every_n_epochs": 10,  # 新增：每隔多少个 epoch 做一次 val nme
            },
        }
    )

    statistics = {"acc": {}}   # 每个 task 结束后的 acc cum
    acc_cum_list = []          # 也保留一个 list 版本

    def collect_acc_cum(test_outputs, cur_task_idx: int):
        acc_cum = []
        for out in test_outputs[: cur_task_idx + 1]:
            acc = next(
                float(v) for k, v in out.items()
                if k.startswith("test_") and not k.startswith("test_nme_")
            )
            acc_cum.append(acc)
        return acc_cum


    from lightning.pytorch.utilities.rank_zero import rank_zero_only

    # @rank_zero_only
    # def log_acc_to_wandb(logger, acc_cum: list[float]):
    #     run = logger.experiment
    #     run.define_metric("statistics/task")
    #     run.define_metric("statistics/acc", step_metric="statistics/task")
    #
    #     for i, acc in enumerate(acc_cum, start=1):
    #         logger.log_metrics({
    #             "statistics/task": i,
    #             "statistics/acc": float(acc),
    #         })

    @rank_zero_only
    def log_acc_to_wandb(logger, cur_task_idx: int, acc_cum: list[float]):
        table = wandb.Table(
            data=[[i + 1, 100*acc] for i, acc in enumerate(acc_cum)],
            columns=["task", "acc"],
        )
        logger.experiment.log({
            "statistics/acc": wandb.plot.line(
                table,
                "task",
                "acc",
                title=f"Final Acc Cum @ Task {cur_task_idx + 1}",
            )
        })

    for task_idx, _ in enumerate(N_CLASS_PER_TASK):
        model.train()
        model.using_distill = task_idx > 0
        model.need_snapshot_old = task_idx == 0
        model.buffer_training = False
        dm.set_current_task(task_idx)
        # use training data, with buffer
        dm.use_buffer = True
        dm.buffer_only_new = False
        dm.train_filter_fn = None

        logger1 = OffsetWandbLogger(
            resume="allow",
            name=f"NME_main_adpmem_nopretrain_20_herding_select_wo_warmup_podnet_cifar100_{'pretrained_' if USE_PRETRAIN_WEIGHTS else ''}task{task_idx}",
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

            #

            # model.backbone.eval()
            # model.backbone.requires_grad_(False)
            # model.classifier.requires_grad_(True)
            dm.use_buffer = True
            dm.buffer_only_new = False
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


        final_test_outputs = final_trainer.test(
            model=model,
            datamodule=dm,
            verbose=False,
            ckpt_path=None,
        )

        cur_acc_cum = collect_acc_cum(final_test_outputs, task_idx)

        statistics["acc"][task_idx] = cur_acc_cum.copy()
        acc_cum_list.append(cur_acc_cum.copy())

        log_acc_to_wandb(final_trainer.logger, task_idx, cur_acc_cum)


        # trainer.validate(model, datamodule=dm)
        wandb.finish()


if __name__ == "__main__":
    test_podnet_cifar100(is_dummy_training=False)

