# banknifty-forensics-omega | FILE-006 | developer task shortcuts (reads pyproject.toml)
PYTHON ?= python3
BNF ?= bnf
CONFIG ?= config/default_config.yaml
STORE_DIR ?= ./store
RUN_DIR ?= ./runs
SEED ?= 42

.PHONY: help install test lint selftest null-lab run clean-cache

help:
	@echo "targets: install test lint selftest null-lab run clean-cache"

install:
	$(PYTHON) -m pip install -e '.[dev,models]'

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src/ tests/

selftest:
	$(BNF) selftest --run-dir $(RUN_DIR) --seed $(SEED)

null-lab:
	$(BNF) null-lab --config $(CONFIG) --store-dir $(STORE_DIR) --run-dir $(RUN_DIR) --seed $(SEED)

run:
	$(BNF) run --config $(CONFIG) --store-dir $(STORE_DIR) --run-dir $(RUN_DIR) --seed $(SEED)

clean-cache:
	rm -rf .pytest_cache .ruff_cache
