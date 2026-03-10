import os.path as osp
import lightning as L
import pytest
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_only
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

# from lightning.pytorch.utilities.rank_zero import rank_zero_only
# @rank_zero_only
# def log_acc_to_wandb(trainer, statistics_summary):
#     for metric_prefix, log_key, title in [
#         ("test_cum", "statistics/acc", "Final Acc"),
#         ("test_nme_cum", "statistics/acc_nme", "Final Acc NME"),
#     ]:
#         data_i = []
#         for task_idx, test_outputs in sorted(statistics_summary.items()):
#             target_key = f"{metric_prefix}/task{task_idx}"
#             for out in test_outputs:
#                 if target_key in out:
#                     acc = round(float(out[target_key]) * 100, 2)
#                     data_i.append([int(task_idx), acc])
#                     break
#         table = wandb.Table(
#             data=data_i,
#             columns=["task", "acc"],
#         )
#         trainer.logger.experiment.log({
#             log_key: wandb.plot.line(
#                 table=table,
#                 x="task",
#                 y="acc",
#                 title=title,
#             )
#         })

from lightning.pytorch.utilities.rank_zero import rank_zero_only
import wandb


def compute_avg_forgetting_curve(statistics_summary, metric_prefix):
    """
    计算每个阶段 i 的平均遗忘率（百分数）:
        F_i = mean_j( max_{t in [j, i]} acc(t, j) - acc(i, j) )

    其中 j 是到阶段 i 为止已经学过的旧任务，默认不包含当前任务 i 自己。
    所以:
        - stage 0 的 forgetting = 0
        - stage i (i>0) 的平均 forgetting 是对任务 0 ~ i-1 求平均

    返回:
        data: [[stage_idx, avg_forgetting_percent], ...]
    """
    stage_ids = sorted(statistics_summary.keys())
    data = []

    for cur_stage in stage_ids:
        # 第一个任务没有 forgetting
        if cur_stage == 0:
            data.append([0, 0.0])
            continue

        forgetting_list = []

        # 只统计旧任务 0 ~ cur_stage-1
        for old_task in stage_ids:
            if old_task >= cur_stage:
                break

            target_key = f"{metric_prefix}/task{old_task}"
            history = []

            # 收集 old_task 从学完自己开始，到当前阶段 cur_stage 为止的 acc 轨迹
            for past_stage in stage_ids:
                if past_stage < old_task:
                    continue
                if past_stage > cur_stage:
                    break

                for out in statistics_summary[past_stage]:
                    if target_key in out:
                        history.append(float(out[target_key]))
                        break

            if not history:
                continue

            cur_acc = history[-1]
            best_acc = max(history)
            forgetting = (best_acc - cur_acc) * 100.0
            forgetting_list.append(forgetting)

        avg_forgetting = (
            round(sum(forgetting_list) / len(forgetting_list), 2)
            if forgetting_list else 0.0
        )
        data.append([int(cur_stage), avg_forgetting])

    return data


# @rank_zero_only
# def log_acc_to_wandb(trainer, statistics_summary):
#     exp = trainer.logger.experiment
#
#     metric_configs = [
#         ("test_cum", "statistics/acc", "Final Acc",
#          "statistics/avg_forgetting", "Avg Forgetting"),
#         ("test_nme_cum", "statistics/acc_nme", "Final Acc NME",
#          "statistics/avg_forgetting_nme", "Avg Forgetting NME"),
#     ]
#
#     for metric_prefix, acc_key, acc_title, fg_key, fg_title in metric_configs:
#         # 1) 每个阶段的 last acc（对角线）
#         acc_data = []
#         for task_idx, test_outputs in sorted(statistics_summary.items()):
#             target_key = f"{metric_prefix}/task{task_idx}"
#             for out in test_outputs:
#                 if target_key in out:
#                     acc = round(float(out[target_key]) * 100, 2)
#                     acc_data.append([int(task_idx), acc])
#                     break
#
#         if acc_data:
#             acc_table = wandb.Table(data=acc_data, columns=["task", "acc"])
#             exp.log({
#                 acc_key: wandb.plot.line(
#                     table=acc_table,
#                     x="task",
#                     y="acc",
#                     title=acc_title,
#                 )
#             })
#
#         # 2) 每个阶段的平均 forgetting
#         fg_data = compute_avg_forgetting_curve(statistics_summary, metric_prefix)
#         if fg_data:
#             fg_table = wandb.Table(data=fg_data, columns=["task", "forgetting"])
#             exp.log({
#                 fg_key: wandb.plot.line(
#                     table=fg_table,
#                     x="task",
#                     y="forgetting",
#                     title=fg_title,
#                 )
#             })

@rank_zero_only
def log_acc_to_wandb(trainer, statistics_summary):
    exp = trainer.logger.experiment

    metric_configs = [
        ("test_cum", "statistics/acc", "Final Acc", "Avg Forgetting"),
        ("test_nme_cum", "statistics/acc_nme", "Final Acc NME", "Avg Forgetting NME"),
    ]

    for metric_prefix, key, acc_title, fg_title in metric_configs:
        # 1) 每个阶段的 last acc（对角线）
        acc_data = []
        for task_idx, test_outputs in sorted(statistics_summary.items()):
            target_key = f"{metric_prefix}/task{task_idx}"
            for out in test_outputs:
                if target_key in out:
                    acc = round(float(out[target_key]) * 100, 2)
                    acc_data.append([int(task_idx), acc])
                    break

        if acc_data:
            acc_table = wandb.Table(data=acc_data, columns=["task", "acc"])
            exp.log({
                key: wandb.plot.line(
                    table=acc_table,
                    x="task",
                    y="acc",
                    title=acc_title,
                )
            })

        # 2) 每个阶段的平均 forgetting
        fg_data = compute_avg_forgetting_curve(statistics_summary, metric_prefix)
        if fg_data:
            fg_table = wandb.Table(data=fg_data, columns=["task", "forgetting"])
            exp.log({
                key: wandb.plot.line(
                    table=fg_table,
                    x="task",
                    y="forgetting",
                    title=fg_title,
                )
            })

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
                "every_n_epochs": 200,  # 新增：每隔多少个 epoch 做一次 val nme
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
        # use training data, with buffer
        dm.use_buffer = True
        dm.buffer_only_new = False
        dm.train_filter_fn = None

        logger1 = OffsetWandbLogger(
            resume="allow",
            name=f"statistic_nme_main_adpmem_nopretrain_20_herding_select_wo_warmup_podnet_cifar100_{'pretrained_' if USE_PRETRAIN_WEIGHTS else ''}task{task_idx}",
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

        test_outputs = final_trainer.test(
            model=model,
            datamodule=dm,
            verbose=False,
            ckpt_path=None,
        )
        statistics_summary[task_idx] = test_outputs
        log_acc_to_wandb(final_trainer, statistics_summary)

        wandb.finish()


if __name__ == "__main__":
    test_podnet_cifar100(is_dummy_training=False)
