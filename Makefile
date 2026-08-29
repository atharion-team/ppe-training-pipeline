.PHONY: help docs serve clean download train
.DEFAULT_GOAL := help
COMPOSE := docker compose -f  docker/docker-compose.yml

# Prefer the venv interpreter, fall back to whatever is on PATH
PYTHON := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,\
          $(if $(wildcard .venv/bin/python),.venv/bin/python,python))
CONFIG ?= data/css-data.yaml
NAME    ?= ppe

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

download: ## Download the dataset (CONFIG=<yaml>, ARGS="--force")
	@$(PYTHON) pipeline/download_dataset.py --data $(CONFIG) $(ARGS)

train: ## Train the detector (NAME=, ARGS="--imgsz 960")
	@$(PYTHON) pipeline/train.py --name $(NAME) $(ARGS)

serve: ## Serve docs live at http://localhost:8000
	@$(COMPOSE) up -d

docs: ## Build the static docs site into site/
	@$(COMPOSE) run --rm docs build

clean: ## Remove generated runs and the built docs site
	@rm -rf site/
