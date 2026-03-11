import os.path as osp

import lightning as L
import pytest
from lightning.pytorch.loggers import WandbLogger

import wandb
from lycil.constants import _EXP_NAME
from lycil.data.hfmodule import HFDataModule
from lycil.learner.ewc import EWC

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

from lightning.pytorch.utilities.rank_zero import rank_zero_only


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


@rank_zero_only
def log_acc_to_wandb(trainer, statistics_summary):
    exp = trainer.logger.experiment

    metric_configs = [
        ("test_cum", "statistics/acc", "Final Acc",
         "statistics/avg_forgetting", "Avg Forgetting"),
        ("test_nme_cum", "statistics/acc_nme", "Final Acc NME",
         "statistics/avg_forgetting_nme", "Avg Forgetting NME"),
    ]

    for metric_prefix, acc_key, acc_title, fg_key, fg_title in metric_configs:
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
                acc_key: wandb.plot.line(
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
                fg_key: wandb.plot.line(
                    table=fg_table,
                    x="task",
                    y="forgetting",
                    title=fg_title,
                )
            })

@pytest.mark.slow
@pytest.mark.runs_on(["cuda"])
@pytest.mark.xdist_group("training")
def test_ewc_cifar100(device: str, is_dummy_training: bool):
    if is_dummy_training:
        DATAPATH, LABEL_COL = CIFAR10_PATH, CIFAR10_LABEL_COL
        N_CLASS_PER_TASK = [1, 1]
        EPOCHS_PER_TASK = 1
    else:
        DATAPATH = "/ppio_net0/datasets/cifar100"
        N_CLASS_PER_TASK = [20, 20, 20, 20, 20]
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
        train_loader_kwargs={"batch_size": 128, "shuffle": True, "num_workers": 10},
        val_loader_kwargs=VAL_LOADER_KWARGS,
        test_loader_kwargs=TEST_LOADER_KWARGS,
        split_map={"train": "train", "val": "test", "test": "test"},
        buffer_kwargs=None,
    )
    model = EWC(
        backbone_args=ConvNetArgs(name="resnet50", pretrained=USE_PRETRAIN_WEIGHTS, cifar=True),
        head="linear",
        per_task_optim_args={
            # for all tasks, use the same optimizer kwargs
            "default": {
                "type": "sgd",
                "lr": 0.1,
                "momentum": 0 if USE_PRETRAIN_WEIGHTS else 0.9,
                "weight_decay": 5e-4,
            },
            0: {
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
            }
        },
        lambda_ewc=1e4,
        fisher_max=0.0001,
        buffer_args=None,

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
                name=f"lambda1e4_nme_warmup_hparms_wd_5e4_ewc_cifar100_task{task_idx}",
                project="lycil",
                log_model=False,
                tags=["ewc", "cifar100"],
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
        log_acc_to_wandb(trainer, statistics_summary)

        wandb.finish()


if __name__ == "__main__":
    test_ewc_cifar100(device="cuda", is_dummy_training=False)
