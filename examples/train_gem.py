import lightning.pytorch as pl
from lightning.pytorch import cli
from lightning_cil.methods.gem import GEM
from data.datamodule_cifar100 import CIFAR100DataModule
from data.datamodule_imagenet import ImageNetDataModule

class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--data.dataset", type=str, default="cifar100", choices=["cifar100", "imagenet"])

    def instantiate_classes(self):
        args = self.parser.parse_args()
        self.datamodule_class = CIFAR100DataModule if getattr(args, "data_dataset", "cifar100") == "cifar100" else ImageNetDataModule
        super().instantiate_classes()

    def run(self):
        pl.seed_everything(self.config.get("seed_everything", None))
        dm = self.datamodule
        model: GEM = self.model
        trainer = self.trainer

        dm.setup()
        num_tasks = dm.num_tasks()
        dm.buffer = model.buffer

        for task_id in range(num_tasks):
            dm.set_task_id(task_id)
            cur, seen = dm.current_classes, dm.seen_classes
            model.expand_head(len(cur))
            model.set_task_info(cur, seen)
            trainer.fit(model, dm)
            trainer.validate(model, dm)
            model.update_memory(datamodule=dm, device=trainer.strategy.root_device)
            model.snapshot_prev_model()

        trainer.test(model, dm)

if __name__ == "__main__":
    CLI(GEM, CIFAR100DataModule, save_config_callback=None, run=True)
