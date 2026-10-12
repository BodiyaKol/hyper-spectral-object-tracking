PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup lint test

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

test:
	$(BIN)/pytest