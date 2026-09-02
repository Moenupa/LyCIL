.PHONY: all help format-check format check test test-slow

check_dirs := src tests

# uvx with fallback, e.g.: 1. `uvx ruff check` 2. `ruff check`
TOOL := $(shell command -v uv >/dev/null 2>&1 && echo "uvx" || echo "")
RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")

all: format check

help:
	@echo "Available targets:"
	@echo "  format-check:  Check code lint/format (no fix)"
	@echo "  format:        Run linter and formatter"
	@echo "  check:         Run type checker"
	@echo "  test:          Run quick tests"
	@echo "  test-slow:     Run tests with real examples (slow)"

format-check:
	$(TOOL) ruff check $(check_dirs)
	$(TOOL) ruff format --check $(check_dirs)

format:
	$(TOOL) ruff check $(check_dirs) --fix
	$(TOOL) ruff format $(check_dirs)

check:
	$(TOOL) ty check $(check_dirs)

test:
	WANDB_MODE=disabled $(RUN) pytest tests -n 8

test-slow:
	RUN_SLOW=1 WANDB_MODE=disabled $(RUN) pytest tests -vv