.PHONY: build install style-check style quality test

check_dirs := src tests

# uvx with fallback, e.g.: 1. `uvx ruff check` 2. `ruff check`
TOOL := $(shell command -v uv >/dev/null 2>&1 && echo "uvx" || echo "")
RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")

build:
	uv build

install:
	$(TOOL) pre-commit install
	$(TOOL) pre-commit run --all-files

# check for style, do nothing
style-check:
	$(TOOL) ruff check $(check_dirs)
	$(TOOL) ruff format --check $(check_dirs)

# check for style, and fix issues
style:
	$(TOOL) ruff check $(check_dirs) --fix
	$(TOOL) ruff format $(check_dirs)

# code quality checks
quality:
	$(TOOL) ty check $(check_dirs)

test:
	WANDB_MODE=disabled $(RUN) pytest tests -n auto --import-mode=importlib

test-slow:
	RUN_SLOW=1 WANDB_MODE=disabled $(RUN) pytest tests --import-mode=importlib