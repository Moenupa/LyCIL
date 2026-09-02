import wandb
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def calc_per_task_forgetting(
    cur_stage: int,
    statistics_summary: dict[int, list[dict[str, float]]],
    metric_prefix: int | str,
) -> list[float]:
    forgetting_list = []

    # Only count old tasks from 0 to cur_stage-1
    for old_task in sorted(statistics_summary.keys()):
        if old_task >= cur_stage:
            break

        target_key = f"{metric_prefix}/task{old_task}"
        history = []

        # Collect the accuracy trajectory for old_task from its completion to the current stage
        for past_stage in sorted(statistics_summary.keys()):
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
    return forgetting_list


def compute_forgetting_curve(
    statistics_summary: dict[int, list[dict[str, float]]], metric_prefix: int | str
) -> dict[int, float]:
    r"""Compute forgetting curve for each stage based on the definition in continual learning.

    :math:`F_i = \frac{1}{i} \sum^{i-1}_{j=0}( \max_{t \in [j, i]} acc(t, j) - acc(i, j) )`

    Args:
        statistics_summary (dict):
            Dictionary containing the statistics for each stage. The keys are stage indices,
            and the values are lists of output dictionaries for each stage.
        metric_prefix (int | str):
            Prefix of the target metric in the output dictionaries, e.g., "test_cum" or "test_nme_cum".

    Returns:
        dict[int, float]: A dictionary mapping each stage index to its corresponding forgetting value.

    """
    stage_ids = sorted(statistics_summary.keys())
    data: dict[int, float] = {0: 0.0}

    for cur_stage in stage_ids:
        if cur_stage == 0:
            continue

        per_task_forgetting = calc_per_task_forgetting(
            cur_stage, statistics_summary, metric_prefix
        )
        forgetting = (
            round(sum(per_task_forgetting) / len(per_task_forgetting), 2)
            if per_task_forgetting
            else 0.0
        )
        data[cur_stage] = forgetting

    return data


@rank_zero_only
def log_statistics_to_wandb(trainer, statistics_summary):
    # wandb logger
    exp = trainer.logger.experiment

    # (metric_prefix, acc_key, acc_title, fg_key, fg_title)
    metric_configs = [
        ("test_cum", "statistics/acc", "Acc", "statistics/forgetting", "Forgetting"),
        (
            "test_nme_cum",
            "statistics/acc_nme",
            "Acc NME",
            "statistics/forgetting_nme",
            "Forgetting NME",
        ),
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
        avg_forgetting = (
            round(sum(v for _, v in fg_data.items()) / len(fg_data), 2)
            if fg_data
            else 0.0
        )

        # log acc / forgetting curves
        for data, key, title, y in (
            (acc_data, acc_key, acc_title, "acc"),
            (fg_data, fg_key, fg_title, "forgetting"),
        ):
            if data:
                exp.log(
                    {
                        key: wandb.plot.line(
                            table=wandb.Table(data=data, columns=["task", y]),
                            x="task",
                            y=y,
                            title=title,
                        )
                    }
                )

        # log average acc / forgetting as scalars
        exp.log(
            {
                f"{acc_key}_avg": avg_acc,
                f"{fg_key}_avg": avg_forgetting,
            }
        )

        results[metric_prefix] = {
            "acc_data": acc_data,
            "forgetting_data": fg_data,
            "avg_acc": avg_acc,
            "avg_forgetting": avg_forgetting,
        }

    # choose best curves between base and NME:
    # max avg_acc, min avg_forgetting
    base, nme = results.get("test_cum"), results.get("test_nme_cum")
    best_acc = (
        base if (base and (not nme or base["avg_acc"] >= nme["avg_acc"])) else nme
    )
    best_fg = (
        base
        if (base and (not nme or base["avg_forgetting"] <= nme["avg_forgetting"]))
        else nme
    )

    # log selected best curves and their scalar summaries
    for best, curve_key, title, y, avg_key, avg_name in (
        (
            best_acc,
            "statistics/max_acc",
            "Max Acc",
            "acc",
            "statistics/max_acc_avg",
            "avg_acc",
        ),
        (
            best_fg,
            "statistics/min_forgetting",
            "Min Forgetting",
            "forgetting",
            "statistics/min_forgetting_avg",
            "avg_forgetting",
        ),
    ):
        if best:
            exp.log(
                {
                    curve_key: wandb.plot.line(
                        table=wandb.Table(data=best[f"{y}_data"], columns=["task", y]),
                        x="task",
                        y=y,
                        title=title,
                    ),
                    avg_key: best[avg_name],
                }
            )
