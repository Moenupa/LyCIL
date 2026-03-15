import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def compute_forgetting_curve(statistics_summary, metric_prefix):
    """
    计算每个阶段 i 的遗忘率（百分数）:
        F_i = mean_j( max_{t in [j, i]} acc(t, j) - acc(i, j) )

    其中 j 是到阶段 i 为止已经学过的旧任务，默认不包含当前任务 i 自己。
    所以:
        - stage 0 的 forgetting = 0
        - stage i (i>0) 的 forgetting 是对任务 0 ~ i-1 求均值

    返回:
        data: [[stage_idx, forgetting_percent], ...]
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

        forgetting = (
            round(sum(forgetting_list) / len(forgetting_list), 2)
            if forgetting_list else 0.0
        )
        data.append([int(cur_stage), forgetting])

    return data


@rank_zero_only
def log_statistics_to_wandb(trainer, statistics_summary):
    # wandb logger
    exp = trainer.logger.experiment

    # (metric_prefix, acc_key, acc_title, fg_key, fg_title)
    metric_configs = [
        ("test_cum", "statistics/acc", "Acc",
         "statistics/forgetting", "Forgetting"),
        ("test_nme_cum", "statistics/acc_nme", "Acc NME",
         "statistics/forgetting_nme", "Forgetting NME"),
    ]

    # cache curves and scalar summaries for later selection
    results = {}

    for metric_prefix, acc_key, acc_title, fg_key, fg_title in metric_configs:
        # diagonal acc curve: stage i -> task i acc
        acc_data = []
        for task_idx, test_outputs in sorted(statistics_summary.items()):
            target_key = f"{metric_prefix}/task{task_idx}"
            for out in test_outputs:
                if target_key in out:
                    acc_data.append([task_idx, round(float(out[target_key]) * 100, 2)])
                    break

        # skip missing metrics (e.g. no NME results)
        if not acc_data:
            continue

        # forgetting curve and corresponding scalar summaries
        fg_data = compute_forgetting_curve(statistics_summary, metric_prefix)
        avg_acc = round(sum(v for _, v in acc_data) / len(acc_data), 2)
        avg_forgetting = round(sum(v for _, v in fg_data) / len(fg_data), 2) if fg_data else 0.0

        # log acc / forgetting curves
        for data, key, title, y in (
            (acc_data, acc_key, acc_title, "acc"),
            (fg_data, fg_key, fg_title, "forgetting"),
        ):
            if data:
                exp.log({
                    key: wandb.plot.line(
                        table=wandb.Table(data=data, columns=["task", y]),
                        x="task",
                        y=y,
                        title=title,
                    )
                })

        # log average acc / forgetting as scalars
        exp.log({
            f"{acc_key}_avg": avg_acc,
            f"{fg_key}_avg": avg_forgetting,
        })

        results[metric_prefix] = {
            "acc_data": acc_data,
            "forgetting_data": fg_data,
            "avg_acc": avg_acc,
            "avg_forgetting": avg_forgetting,
        }

    # choose best curves between base and NME:
    # max avg_acc, min avg_forgetting
    base, nme = results.get("test_cum"), results.get("test_nme_cum")
    best_acc = base if (base and (not nme or base["avg_acc"] >= nme["avg_acc"])) else nme
    best_fg = base if (base and (not nme or base["avg_forgetting"] <= nme["avg_forgetting"])) else nme

    # log selected best curves and their scalar summaries
    for best, curve_key, title, y, avg_key, avg_name in (
        (best_acc, "statistics/max_acc", "Max Acc", "acc", "statistics/max_acc_avg", "avg_acc"),
        (best_fg, "statistics/min_forgetting", "Min Forgetting", "forgetting", "statistics/min_forgetting_avg", "avg_forgetting"),
    ):
        if best:
            exp.log({
                curve_key: wandb.plot.line(
                    table=wandb.Table(data=best[f"{y}_data"], columns=["task", y]),
                    x="task",
                    y=y,
                    title=title,
                ),
                avg_key: best[avg_name],
            })


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

# TODO: NPU support
from lightning.pytorch.accelerators import CUDAAccelerator
def resolve_num_devices(accelerator: str, devices) -> int:
    """Resolve the number of devices used for LR scaling."""
    if accelerator not in {"cuda", "gpu"}:
        return 1

    if devices in (None, "auto", -1, "-1"):
        return CUDAAccelerator.auto_device_count()

    if isinstance(devices, int):
        return devices

    if isinstance(devices, (list, tuple)):
        return len(devices)

    if isinstance(devices, str):
        s = devices.replace(" ", "")
        if s.isdigit():
            return int(s)
        return len([x for x in s.split(",") if x])

    return 1


def build_per_task_optim_sched_args(
    num_tasks: int,
    epochs_per_task: int,
    use_pretrain_weights: bool,
):
    per_task_optim_args = {
        "default": {
            "type": "sgd",
            "lr": 0.1,
            "momentum": 0 if use_pretrain_weights else 0.9,
            "weight_decay": 5e-4,
        },
    }

    per_task_sched_args = {
        "default": {
            "type": "linear_warmup_cosine_annealing",
            "warmup_epochs": 0 if epochs_per_task == 1 else 10,
            "max_epochs": epochs_per_task,
        },
    }

    # 从第 1 个任务开始覆盖 default
    for task_idx in range(1, num_tasks):
        per_task_optim_args[task_idx] = {
            "type": "adam",
            "lr": 1e-3,
            "weight_decay": 5e-4,
        }
        per_task_sched_args[task_idx] = {
            # "type": "cosine_annealing",
            # "T_max": epochs_per_task,
            "type": "linear_warmup_cosine_annealing",
            "warmup_epochs": 0 if epochs_per_task == 1 else 10,
            "max_epochs": epochs_per_task,
        }

    return per_task_optim_args, per_task_sched_args