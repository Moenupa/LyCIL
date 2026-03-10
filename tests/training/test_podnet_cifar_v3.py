import json
import math
import os.path as osp
from pathlib import Path
from typing import Any

import lightning as L
import pytest
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from lycil.backbone import ConvNetArgs
from lycil.constants import _EXP_NAME
from lycil.data.hfmodule import HFDataModule
from lycil.learner.podnet import PODNet


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


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _extract_validate_metrics(result: dict[str, Any]) -> dict[str, Any]:
    acc_key = next(
        (k for k in result.keys() if k.startswith("val_") and not k.startswith("val_nme_")),
        None,
    )
    nme_key = next((k for k in result.keys() if k.startswith("val_nme_")), None)

    suffix = acc_key[len("val_"):] if acc_key is not None else None
    return {
        "suffix": suffix,
        "acc_key": acc_key,
        "acc": _to_float(result.get(acc_key)) if acc_key is not None else None,
        "nme_key": nme_key,
        "nme_acc": _to_float(result.get(nme_key)) if nme_key is not None else None,
        "raw": {k: _to_float(v) for k, v in result.items()},
    }


def collect_last_epoch_eval(
    trainer: L.Trainer, model: PODNet, dm: HFDataModule, task_idx: int
) -> dict[str, Any]:
    results = trainer.validate(model, datamodule=dm, verbose=False)
    per_task_metrics = []
    for eval_task_idx, result in enumerate(results):
        metrics = _extract_validate_metrics(result)
        metrics["eval_task_idx"] = eval_task_idx
        per_task_metrics.append(metrics)

    return {
        "after_task": task_idx,
        "metrics": per_task_metrics,
    }


def _build_metric_matrix(
    task_eval_history: list[dict[str, Any]], field: str
) -> tuple[list[list[float]], list[str]]:
    n_tasks = len(task_eval_history)
    matrix = [[float("nan") for _ in range(n_tasks)] for _ in range(n_tasks)]
    task_names = [f"task{idx}" for idx in range(n_tasks)]

    for row in task_eval_history:
        after_task = row["after_task"]
        for item in row["metrics"]:
            eval_task_idx = item["eval_task_idx"]
            metric_value = item.get(field)
            if eval_task_idx < n_tasks and metric_value is not None:
                matrix[after_task][eval_task_idx] = metric_value
            if item.get("suffix"):
                task_names[eval_task_idx] = item["suffix"]

    return matrix, task_names


def _nanmean(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def compute_avg_accuracy_curve(matrix: list[list[float]]) -> list[float]:
    curve = []
    for after_task, row in enumerate(matrix):
        curve.append(_nanmean(row[: after_task + 1]))
    return curve


def compute_forgetting_curve(matrix: list[list[float]]) -> list[float]:
    n_tasks = len(matrix)
    curve: list[float] = []
    for after_task in range(n_tasks):
        if after_task == 0:
            curve.append(0.0)
            continue

        forgetting_values = []
        for eval_task in range(after_task):
            current = matrix[after_task][eval_task]
            if math.isnan(current):
                continue

            history = [matrix[past_task][eval_task] for past_task in range(eval_task, after_task)]
            history = [v for v in history if not math.isnan(v)]
            if not history:
                continue

            forgetting_values.append(max(history) - current)

        curve.append(_nanmean(forgetting_values) if forgetting_values else 0.0)

    return curve


def compute_final_forgetting_per_task(
    matrix: list[list[float]], task_names: list[str]
) -> list[dict[str, Any]]:
    if not matrix:
        return []

    final_row_idx = len(matrix) - 1
    summary = []
    for eval_task in range(final_row_idx):
        current = matrix[final_row_idx][eval_task]
        if math.isnan(current):
            continue

        history = [matrix[past_task][eval_task] for past_task in range(eval_task, final_row_idx)]
        history = [v for v in history if not math.isnan(v)]
        if not history:
            continue

        summary.append(
            {
                "task_idx": eval_task,
                "task_name": task_names[eval_task],
                "best_past_acc": max(history),
                "final_acc": current,
                "forgetting": max(history) - current,
            }
        )
    return summary

def log_summary_to_wandb(
    *,
    base_run_name: str,
    project: str,
    group: str,
    tags: list[str],
    task_eval_history: list[dict[str, Any]],
    output_dir: str,
) -> None:
    acc_matrix, task_names = _build_metric_matrix(task_eval_history, field="acc")
    nme_matrix, _ = _build_metric_matrix(task_eval_history, field="nme_acc")

    avg_acc_curve = compute_avg_accuracy_curve(acc_matrix)
    forgetting_curve = compute_forgetting_curve(acc_matrix)
    final_forgetting = compute_final_forgetting_per_task(acc_matrix, task_names)

    has_nme = any(
        item.get("nme_acc") is not None
        for row in task_eval_history
        for item in row["metrics"]
    )
    avg_nme_curve = compute_avg_accuracy_curve(nme_matrix) if has_nme else []
    nme_forgetting_curve = compute_forgetting_curve(nme_matrix) if has_nme else []
    final_nme_forgetting = (
        compute_final_forgetting_per_task(nme_matrix, task_names) if has_nme else []
    )

    summary_payload = {
        "task_names": task_names,
        "task_eval_history": task_eval_history,
        "classifier": {
            "accuracy_matrix": acc_matrix,
            "avg_accuracy_curve": avg_acc_curve,
            "forgetting_curve": forgetting_curve,
            "final_avg_accuracy": avg_acc_curve[-1] if avg_acc_curve else None,
            "final_avg_forgetting": forgetting_curve[-1] if forgetting_curve else None,
            "final_forgetting_per_task": final_forgetting,
        },
    }

    if has_nme:
        summary_payload["nme"] = {
            "accuracy_matrix": nme_matrix,
            "avg_accuracy_curve": avg_nme_curve,
            "forgetting_curve": nme_forgetting_curve,
            "final_avg_accuracy": avg_nme_curve[-1] if avg_nme_curve else None,
            "final_avg_forgetting": nme_forgetting_curve[-1] if nme_forgetting_curve else None,
            "final_forgetting_per_task": final_nme_forgetting,
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_path / f"{base_run_name}_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    summary_run = wandb.init(
        project=project,
        group=group,
        name=f"{base_run_name} summary",
        tags=tags + ["summary"],
        job_type="summary",
        reinit=True,
    )

    # 让 wandb 用 after_task 作为横轴
    summary_run.define_metric("summary/after_task")
    summary_run.define_metric("summary/*", step_metric="summary/after_task")

    # 原始长表：task 间评测结果
    matrix_table = wandb.Table(
        columns=["after_task", "eval_task", "task_name", "acc", "nme_acc"]
    )
    for row in task_eval_history:
        after_task = row["after_task"]
        for item in row["metrics"]:
            matrix_table.add_data(
                after_task,
                item["eval_task_idx"],
                item.get("suffix") or f"task{item['eval_task_idx']}",
                item.get("acc"),
                item.get("nme_acc"),
            )

    # 原始长表：summary 曲线
    avg_curve_table = wandb.Table(
        columns=["after_task", "avg_acc", "forgetting", "avg_nme_acc", "nme_forgetting"]
    )
    for after_task in range(len(task_eval_history)):
        avg_curve_table.add_data(
            after_task,
            avg_acc_curve[after_task] if after_task < len(avg_acc_curve) else None,
            forgetting_curve[after_task] if after_task < len(forgetting_curve) else None,
            avg_nme_curve[after_task] if has_nme and after_task < len(avg_nme_curve) else None,
            nme_forgetting_curve[after_task]
            if has_nme and after_task < len(nme_forgetting_curve)
            else None,
        )

    summary_run.log(
        {
            "summary/final_eval_matrix": matrix_table,
            "summary/curve_table": avg_curve_table,
        }
    )

    # 按 after_task 逐点记录 classifier 曲线
    for after_task in range(len(task_eval_history)):
        log_dict = {
            "summary/after_task": after_task,
            "summary/classifier_avg_accuracy": avg_acc_curve[after_task],
            "summary/classifier_forgetting": forgetting_curve[after_task],
        }

        # 每个 eval task 一条准确率曲线
        for eval_task_idx, task_name in enumerate(task_names):
            value = acc_matrix[after_task][eval_task_idx]
            if not math.isnan(value):
                log_dict[f"summary/classifier_acc/{task_name}"] = value

        if has_nme:
            log_dict["summary/nme_avg_accuracy"] = avg_nme_curve[after_task]
            log_dict["summary/nme_forgetting"] = nme_forgetting_curve[after_task]
            for eval_task_idx, task_name in enumerate(task_names):
                value = nme_matrix[after_task][eval_task_idx]
                if not math.isnan(value):
                    log_dict[f"summary/nme_acc/{task_name}"] = value

        summary_run.log(log_dict)

    # final summary 标量，方便在 run overview / compare 里直接看
    if avg_acc_curve:
        summary_run.summary["summary/final_classifier_avg_accuracy"] = avg_acc_curve[-1]
    if forgetting_curve:
        summary_run.summary["summary/final_classifier_avg_forgetting"] = forgetting_curve[-1]

    for item in final_forgetting:
        task_name = item["task_name"]
        summary_run.summary[f"summary/final_forgetting/{task_name}"] = item["forgetting"]
        summary_run.summary[f"summary/final_acc/{task_name}"] = item["final_acc"]
        summary_run.summary[f"summary/best_past_acc/{task_name}"] = item["best_past_acc"]

    if has_nme:
        if avg_nme_curve:
            summary_run.summary["summary/final_nme_avg_accuracy"] = avg_nme_curve[-1]
        if nme_forgetting_curve:
            summary_run.summary["summary/final_nme_avg_forgetting"] = nme_forgetting_curve[-1]

        for item in final_nme_forgetting:
            task_name = item["task_name"]
            summary_run.summary[f"summary/nme_final_forgetting/{task_name}"] = item["forgetting"]
            summary_run.summary[f"summary/nme_final_acc/{task_name}"] = item["final_acc"]
            summary_run.summary[f"summary/nme_best_past_acc/{task_name}"] = item["best_past_acc"]

    summary_artifact = wandb.Artifact(f"{base_run_name}_summary", type="summary")
    summary_artifact.add_file(str(summary_json_path))
    summary_run.log_artifact(summary_artifact)
    summary_run.finish()

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
        EPOCHS_PER_TASK = 20
        EPOCHS_PER_TASK_MEMORY = 10
        USE_PRETRAIN_WEIGHTS = False
        BUFFER_SIZE_PER_CLASS = 10
    if not osp.exists(DATAPATH):
        pytest.skip("Data path does not exist.")
        return

    L.seed_everything(42)
    total_buffer_size = BUFFER_SIZE_PER_CLASS * sum(N_CLASS_PER_TASK)
    project_name = "lycil"
    run_tags = ["podnet", "cifar100"] + ["pretrained" if USE_PRETRAIN_WEIGHTS else "random_init"]
    base_run_name = (
        f"NME_main_adpmem_nopretrain_20_herding_select_wo_warmup_podnet_cifar100_"
        f"{'pretrained_' if USE_PRETRAIN_WEIGHTS else ''}final"
    )

    dm = HFDataModule(
        DATAPATH,
        transform_name=osp.basename(DATAPATH),
        num_classes_per_task=N_CLASS_PER_TASK,
        label_column_name=LABEL_COL,
        train_loader_kwargs={"batch_size": 128, "shuffle": True, "num_workers": 10},
        val_loader_kwargs={"batch_size": 128, "shuffle": False, "num_workers": 10},
        test_loader_kwargs={"batch_size": 128, "shuffle": False, "num_workers": 10},
        split_map={"train": "train", "val": "test", "test": "test"},
        buffer_kwargs={"mem_size": total_buffer_size},
    )

    model = PODNet(
        backbone_args=ConvNetArgs(name="resnet50", pretrained=USE_PRETRAIN_WEIGHTS, cifar=True),
        head="cosine",
        per_task_optim_args={
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
                "every_n_epochs": 10,
            },
        },
    )

    task_eval_history: list[dict[str, Any]] = []

    for task_idx, _ in enumerate(N_CLASS_PER_TASK):
        model.train()
        model.using_distill = task_idx > 0
        model.need_snapshot_old = task_idx == 0
        model.buffer_training = False

        dm.set_current_task(task_idx)
        dm.use_buffer = True
        dm.buffer_only_new = False
        dm.train_filter_fn = None

        task_run_name = f"{base_run_name}_task{task_idx}"
        logger1 = OffsetWandbLogger(
            resume="allow",
            name=task_run_name,
            project=project_name,
            log_model=False,
            tags=run_tags,
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

            dm.use_buffer = True
            dm.buffer_only_new = False
            dm.train_filter_fn = lambda e: False

            logger2 = OffsetWandbLogger(
                resume="allow",
                step_offset=trainer1.global_step,
                epoch_offset=trainer1.current_epoch + 1,
                name=task_run_name,
                project=project_name,
                log_model=False,
                tags=run_tags,
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

        task_eval_history.append(
            collect_last_epoch_eval(
                trainer=final_trainer,
                model=model,
                dm=dm,
                task_idx=task_idx,
            )
        )
        wandb.finish()

    log_summary_to_wandb(
        base_run_name=base_run_name,
        project=project_name,
        group=_EXP_NAME,
        tags=run_tags,
        task_eval_history=task_eval_history,
        output_dir="./wandb_summaries",
    )


if __name__ == "__main__":
    test_podnet_cifar100(is_dummy_training=False)