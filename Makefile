PYTHON ?= python3
PIP := $(PYTHON) -m pip
export PYTHONPATH := src

.PHONY: install test lint profile generate evaluate run-all clean

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m compileall -q src

profile:
	$(PYTHON) -m synthdata profile

generate:
	$(PYTHON) -m synthdata generate --generator copula --rows 2000

evaluate:
	$(PYTHON) -m synthdata evaluate --source artifacts/source.csv --synthetic artifacts/synthetic-copula.csv

run-all:
	$(PYTHON) -m synthdata run-all

clean:
	rm -rf artifacts .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
