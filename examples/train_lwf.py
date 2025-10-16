import lightning.pytorch as pl
from lightning.pytorch import cli

from lightning_cil.data.cil_datamodule import BaseCILDataModule
from lightning_cil.methods.lwf import LWF


class CLI(cli.LightningCLI):
    model: LWF
    datamodule: BaseCILDataModule

    def instantiate_classes(self):
        super().instantiate_classes()

        # init first task to pass sanity checks
        self.model.expand_head(self.datamodule.num_class_per_task)

    def run(self):
        pl.seed_everything(self.config.get("seed_everything", None))
        dm = self.datamodule
        model: LWF = self.model
        trainer = self.trainer

        for task_id in range(dm.num_tasks):
            dm.set_task(task_id)
            cur = dm.classes_current
            seen = dm.classes_seen
            model.set_task_info(cur, seen)
            # first task already expanded in instantiate_classes()
            model.expand_head(0 if len(cur) == 0 else len(cur))

            trainer.fit(model=model, datamodule=dm)
            trainer.validate(model=model, datamodule=dm)

            # LwF: snapshot previous model; no memory update
            model.snapshot_prev_model()

        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    CLI(
        LWF,
        pl.LightningDataModule,
        save_config_callback=None,
        run=True,
        subclass_mode_data=True,
    )
