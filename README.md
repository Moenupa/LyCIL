# lightning_cil

A clean, [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) (2.x) continual/incremental learning (CIL) library inspired by [PyCIL](https://github.com/LAMDA-CL/PyCIL), with up-to-date methods, NPU support, modern interface, and ready-to-run CLI examples. 

## Features

- Modular **LightningModule** implementations for `iCaRL` and `LUCIR` that inherit from a common `BaseIncremental`.
- `ExemplarBuffer` with **herding** selection, per-class quotas, and persistence.
- `CosineClassifier` with dynamic expansion (used by LUCIR), and a standard Linear head (used by iCaRL by default; switchable).
- **CIFAR-100** and **ImageNet** LightningDataModules with task-by-task class splits.
- Custom **LightningCLI** subclass that orchestrates multi-task training loops.
- Concise, English docstrings and comments.

## Quick Start

We highly recommend using `uv` to manage the environment.

```sh
# uv venv                           # create virtual env
uv sync --extra lightning           # for cuda lightning
uv sync --extra lightning-npu       # for npu lightning
```

<details><summary>Alternatively, install via pip or conda:</summary>

```sh
# conda or pip, with optional dependencies for lightning
pip install -e ".[lightning]"       # 'lightning', or 'lightning-npu'
```

</details>

Use [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) commands to **train**, **validate**, **test**, or **predict** with continual learning models. For example, to train learning without forgetting (LwF) on CIFAR-100:

```sh
# valid commands: {fit,validate,test,predict}
python examples/lwf-cli.py fit -c configs/smoketest_lwf.yml
# or pass individual config files
python examples/lwf-cli.py fit --trainer configs/trainer/smoketest.yml --model configs/model/lwf.yml --data configs/data/cifar100.yml
```