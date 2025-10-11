# lightning_cil

A clean, PyTorch Lightning (>2.0) continual/incremental learning (CIL) library with standard iCaRL and LUCIR implementations, exemplar memory, cosine classifier, and ready-to-run CLI examples.

## Features
- Modular **LightningModule** implementations for `iCaRL` and `LUCIR` that inherit from a common `BaseIncremental`.
- `ExemplarBuffer` with **herding** selection, per-class quotas, and persistence.
- `CosineClassifier` with dynamic expansion (used by LUCIR), and a standard Linear head (used by iCaRL by default; switchable).
- **CIFAR-100** and **ImageNet** LightningDataModules with task-by-task class splits.
- Custom **LightningCLI** subclass that orchestrates multi-task training loops.
- Concise, English docstrings and comments.

## Quickstart (CIFAR-100 + iCaRL)

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Train iCaRL on CIFAR-100 with 10-class increments (example)
python examples/train_icarl.py     --data.root ./data_cache     --data.increment 10     --data.num_workers 4     --model.backbone_name resnet18     --model.head linear     --model.lr 0.1 --model.weight_decay 1e-4     --model.mem_size 2000     --trainer.max_epochs 70     --trainer.accelerator auto --trainer.devices auto
```

## LUCIR (CIFAR-100)

```bash
python examples/train_lucir.py     --data.root ./data_cache     --data.increment 10     --model.backbone_name resnet18     --model.head cosine     --model.lr 0.1 --model.weight_decay 1e-4     --model.mem_size 2000     --model.lucir_margin 0.5 --model.lucir_distill_T 2.0     --trainer.max_epochs 70
```

> For ImageNet, point `--data.root` to an ImageNet-style folder:
> `root/train/<class>/*.JPEG`, `root/val/<class>/*.JPEG`.

## Project Tree
```
lightning_cil/
├── README.md
├── requirements.txt
├── data/
│   ├── __init__.py
│   ├── buffer.py
│   ├── datamodule_cifar100.py
│   └── datamodule_imagenet.py
├── models/
│   ├── backbone/
│   │   └── resnet.py
│   └── classifier/
│       └── cosine_classifier.py
├── lightning_cil/
│   ├── __init__.py
│   └── methods/
│       ├── __init__.py
│       ├── base.py
│       ├── icarl.py
│       └── lucir.py
├── utils/
│   ├── __init__.py
│   └── metrics.py
└── examples/
    ├── train_icarl.py
    └── train_lucir.py
```

## Notes
- This code aims for **complete** functionality with standard losses and training flows (not pedagogical simplifications), while keeping the interfaces clean.
- For research use, tune `--trainer.max_epochs`, LR milestones/schedulers, and augmentation settings to match specific papers.
- Reproducibility: set `--seed_everything <seed>` if needed.
