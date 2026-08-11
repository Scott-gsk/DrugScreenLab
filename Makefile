CONDA ?= conda
CONDA_ENV := drugscreening-gpu
CONDA_RUN = $(CONDA) run --no-capture-output -n $(CONDA_ENV)

.PHONY: env-check test data-check gpu-check validate

env-check:
	$(CONDA_RUN) python scripts/check_environment.py

test:
	$(CONDA_RUN) python scripts/check_environment.py
	$(CONDA_RUN) python -m pytest --capture=no

data-check:
	$(CONDA_RUN) python scripts/check_environment.py
	PYTHONPATH=src $(CONDA_RUN) python -m drug_screen.data.registry --root data

gpu-check:
	$(CONDA_RUN) python scripts/check_environment.py --require-torch --require-cuda

validate: data-check
