import os

import lightning as L
from lightning.pytorch.loggers import WandbLogger

import wandb
from lycil.data.hfmodule import HFDataModule
from lycil.learner.icarl import ICaRL


def main():
    BUFFER_SIZE_PER_CLASS = 20
    NUM_CLASSES_PER_TASK = [50, 10, 10, 10, 10]
    EPOCHS_PER_TASK = 80
    USE_PRETRAIN_WEIGHTS = os.environ.get("PRETRAIN", "1") == "1"

    L.seed_everything(42)
    dm = HFDataModule(
        "data/cifar100",
        transform_name="cifar100",
        num_classes_per_task=NUM_CLASSES_PER_TASK,
        label_column_name="fine_label",  # 100 classes
        train_loader_kwargs={"batch_size": 64, "shuffle": True, "num_workers": 10},
        val_loader_kwargs={"shuffle": False, "num_workers": 10},
        test_loader_kwargs={"shuffle": False, "num_workers": 10},
        split_map={"val": "test"},
        buffer_kwargs={"mem_size_per_class": BUFFER_SIZE_PER_CLASS},
    )
    model = ICaRL(
        backbone_args={
            "name": "resnet50",
            "pretrained": USE_PRETRAIN_WEIGHTS,
            "cifar": True,
        },
        head="linear",
        per_task_optim_args={
            # for all tasks, use the same optimizer kwargs
            -1: {
                "type": "sgd",
                "lr": 0.3,
                "weight_decay": 5e-4,
            },
        },
        per_task_sched_args={
            # for all tasks, use the same scheduler kwargs
            -1: {
                "type": "linear_warmup_cosine_annealing",
                "warmup_epochs": 10,
                "max_epochs": EPOCHS_PER_TASK,
            }
        },
        distill_T=1.0,
        distill_lambda=0.1,
    )

    for task_idx, _ in enumerate(NUM_CLASSES_PER_TASK):
        dm.set_current_task(task_idx)

        trainer = L.Trainer(
            max_epochs=EPOCHS_PER_TASK,
            sync_batchnorm=True,
            enable_checkpointing=False,
            enable_progress_bar=False,
            precision="16-mixed",
            logger=WandbLogger(
                name=f"icarl_cifar100_{'pretrained_' if USE_PRETRAIN_WEIGHTS else ''}task{task_idx}",
                project="lycil",
                log_model=False,
                tags=["icarl", "cifar100"]
                + ["pretrained" if USE_PRETRAIN_WEIGHTS else "random_init"],
            ),
            check_val_every_n_epoch=10,
            log_every_n_steps=1000,
        )
        trainer.fit(model, datamodule=dm)
        trainer.validate(model, datamodule=dm)

        wandb.finish()


if __name__ == "__main__":
    main()
