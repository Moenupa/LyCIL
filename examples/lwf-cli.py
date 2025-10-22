import lightning as L
from lightning.pytorch.cli import LightningCLI

from lycil.data.cil_datamodule import BaseCILDataModule
from lycil.methods.lwf import LWF


class CLI(LightningCLI):
    """
    Train a model using the LwF (`Learning without Forgetting`_) method.

    Examples::

        python examples/lwf-cli.py fit -c configs/smoketest_lwf.yml
        python examples/lwf-cli.py fit --trainer configs/trainer/smoketest.yml --model configs/model/lwf.yml --data configs/data/cifar100.yml

    .. _Learning without Forgetting:
        https://arxiv.org/abs/1606.09282
    """

    model: LWF
    datamodule: BaseCILDataModule

    def fit(self, **kwargs):
        L.seed_everything(self.config.get("seed_everything", None))
        dm = self.datamodule
        model: LWF = self.model
        trainer = self.trainer

        for task_id in range(dm.num_tasks):
            dm.set_task(task_id)
            cur = dm.classes_current
            seen = dm.classes_seen
            model.set_task_info(cur, seen)
            model.expand_head(len(cur))

            trainer.fit(model=model, datamodule=dm)
            trainer.validate(model=model, datamodule=dm)

            # LwF: snapshot previous model; no memory update
            model.snapshot_prev_model()

        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    CLI(
        LWF,
        BaseCILDataModule,
        save_config_callback=None,
        run=True,
        subclass_mode_data=True,
    )
