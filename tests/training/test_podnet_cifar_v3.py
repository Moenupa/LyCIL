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


import re


def extract_task_accs(test_results):
    """
    从 trainer.test() 的返回结果里提取:
    - task_accs: 当前 task 结束后，各已见任务上的 acc
    - acc_cum: task_accs 的平均值
    """
    if isinstance(test_results, list) and len(test_results) > 0:
        merged = {}
        for x in test_results:
            merged.update(x)
    else:
        merged = test_results or {}

    # 只取 test_ 开头、但排除 test_nme_
    acc_items = []
    for k, v in merged.items():
        if k.startswith("test_") and not k.startswith("test_nme_"):
            m = re.search(r"(\d+)$", k)
            idx = int(m.group(1)) if m else 10**9
            if hasattr(v, "item"):
                v = v.item()
            acc_items.append((idx, float(v)))

    acc_items.sort(key=lambda x: x[0])
    task_accs = [v for _, v in acc_items]
    acc_cum = sum(task_accs) / len(task_accs) if len(task_accs) > 0 else None
    return task_accs, acc_cum


def log_summary_to_wandb(logger, task_idx, task_accs, acc_cum, acc_cum_history):
    run = logger.experiment
    run_name = getattr(logger, "_name", None) or getattr(logger, "name", None) or "run"

    summary_text = "\n".join([
        f"after task {task_idx}",
        f"task_accs: {[round(x, 4) for x in task_accs]}",
        f"acc_cum: {round(acc_cum, 4) if acc_cum is not None else None}",
        f"acc_cum_history: {[round(x, 4) for x in acc_cum_history]}",
    ])

    # 写到 wandb summary
    run.summary[f"{run_name} summary"] = summary_text
    run.summary[f"{run_name}/last_task_accs"] = [round(x, 4) for x in task_accs]
    run.summary[f"{run_name}/acc_cum_history"] = [round(x, 4) for x in acc_cum_history]

    # 也顺手 log 一下标量，方便看曲线
    log_dict = {
        f"{run_name}/final_acc_cum": acc_cum,
    }
    for i, acc in enumerate(task_accs):
        log_dict[f"{run_name}/task_{i}_acc"] = acc

    run.log(log_dict)

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
        EPOCHS_PER_TASK = 10
        EPOCHS_PER_TASK_MEMORY = 10
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
                "enable": True,
                "topk": 1,
                "dynamic_old": True,
                "dynamic_new": True,
                "every_n_epochs": 10,  # 新增：每隔多少个 epoch 做一次 val nme
            },
        }
    )

    final_task_accs = {}  # {task_idx: [acc_task0, acc_task1, ...]}
    acc_cum_history = []  # [task0后的acc_cum, task1后的acc_cum, ...]

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
        final_logger = logger1
        final_trainer = trainer1

        if task_idx > 0:
            model.using_distill = True
            model.buffer_training = True
            model.need_snapshot_old = True

            # model.backbone.eval()
            # model.backbone.requires_grad_(False)
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
            final_logger = logger2
            final_trainer = trainer2
            # model.backbone.requires_grad_(True)

        # ===== 每个 task 训练完成后的最终 eval =====
        test_results = final_trainer.test(model, datamodule=dm, verbose=False)
        task_accs, acc_cum = extract_task_accs(test_results)

        final_task_accs[task_idx] = task_accs
        if acc_cum is not None:
            acc_cum_history.append(acc_cum)

        log_summary_to_wandb(
            logger=final_logger,
            task_idx=task_idx,
            task_accs=task_accs,
            acc_cum=acc_cum,
            acc_cum_history=acc_cum_history,
        )

        # trainer.validate(model, datamodule=dm)
        wandb.finish()


if __name__ == "__main__":
    test_podnet_cifar100(is_dummy_training=False)

