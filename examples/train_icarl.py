import os
import torch
import lightning.pytorch as pl
from lightning.pytorch import cli
from lightning_cil.methods.icarl import ICaRL
from data.datamodule_cifar100 import CIFAR100DataModule
from data.datamodule_imagenet import ImageNetDataModule

class CLI(cli.LightningCLI):
    """CIL CLI: orchestrates multi-task loop using LightningCLI interface.

    Usage:
      python examples/train_icarl.py --help
    """
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.increment", "model.num_classes_total", compute_fn=lambda inc: getattr(self.datamodule, "num_classes_total", 100))
        parser.add_argument("--data.dataset", type=str, default="cifar100", choices=["cifar100", "imagenet"],
                            help="Choose dataset. Controls the DataModule class.")

    def instantiate_classes(self):
        # Choose datamodule by flag
        datamodule_class = CIFAR100DataModule if ("--data.dataset" in self.parser.parse_args()) else CIFAR100DataModule
        args = self.parser.parse_args()
        if getattr(args, "data_dataset", "cifar100") == "imagenet":
            self.datamodule_class = ImageNetDataModule
        else:
            self.datamodule_class = CIFAR100DataModule
        # Let parent instantiate model, trainer, datamodule
        super().instantiate_classes()

    def run(self):
        pl.seed_everything(self.config.get("seed_everything", None))
        dm = self.datamodule
        model: ICaRL = self.model
        trainer = self.trainer

        dm.setup()
        num_tasks = dm.num_tasks()
        dm.buffer = model.buffer  # share memory

        seen = []
        for task_id in range(num_tasks):
            dm.set_task_id(task_id)
            cur = dm.current_classes
            seen = dm.seen_classes
            is_first = (task_id == 0)

            # expand head and set task info
            model.expand_head(len(cur))
            model.set_task_info(current_classes=cur, seen_classes=seen)

            # fit this task
            trainer.fit(model=model, datamodule=dm)
            trainer.validate(model=model, datamodule=dm)

            # update memory and snapshot prev model
            model.update_memory(datamodule=dm, device=trainer.strategy.root_device)
            model.snapshot_prev_model()

        # final test on last seen classes
        trainer.test(model=model, datamodule=dm)

if __name__ == "__main__":
    CLI(ICaRL, CIFAR100DataModule, save_config_callback=None, run=True)
