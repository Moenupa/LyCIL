import lightning.pytorch as pl
from lightning.pytorch import cli

from lightning_cil.data.cil_datamodule import BaseCILDataModule
from lightning_cil.methods.icarl import ICaRL


class CLI(cli.LightningCLI):
    """
    Train a model using the iCaRL (`Incremental Classifier and Representation Learning`_) method.

    Examples::

        python examples/icarl-cli.py fit --trainer configs/trainer/smoketest.yml --model configs/model/icarl.yml --data configs/data/cifar100.yml

    .. _Incremental Classifier and Representation Learning:
        https://arxiv.org/abs/1606.09282
    """

    model: ICaRL
    datamodule: BaseCILDataModule

    def fit(self, **kwargs):
        pl.seed_everything(self.config.get("seed_everything", None))
        dm = self.datamodule
        model = self.model
        trainer = self.trainer

        for task_id in range(dm.num_tasks):
            dm.set_task(task_id)
            cur = dm.classes_current
            seen = dm.classes_seen
            model.set_task_info(cur, seen)
            model.expand_head(len(cur))

            trainer.fit(model=model, datamodule=dm)
            trainer.validate(model=model, datamodule=dm)

            # iCarl: snapshot previous model; NME memory
            model.update_memory(dm)
            model.snapshot_prev_model()

        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    CLI(
        ICaRL,
        BaseCILDataModule,
        save_config_callback=None,
        run=True,
        subclass_mode_data=True,
    )
