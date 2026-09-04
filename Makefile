.PHONY: help docs serve clean download train evaluate track associate run
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

evaluate: ## Evaluate a trained model (NAME=, WEIGHTS=<path> overrides, ARGS="--split val")
	@$(PYTHON) pipeline/evaluate.py --weights $(if $(WEIGHTS),$(WEIGHTS),runs/train/$(NAME)/weights/best.pt) $(ARGS)

track: ## Track persons in a video (NAME=, WEIGHTS=<path> overrides, SOURCE=<path>, ARGS="--show")
	@$(PYTHON) pipeline/track.py --weights $(if $(WEIGHTS),$(WEIGHTS),runs/train/$(NAME)/weights/best.pt) --source $(SOURCE) $(ARGS)

associate: ## Track + associate PPE to workers (NAME=, WEIGHTS=<path>, SOURCE=<path>, ARGS="--min-containment 0.6")
	@$(PYTHON) pipeline/associate.py --weights $(if $(WEIGHTS),$(WEIGHTS),runs/train/$(NAME)/weights/best.pt) --source $(SOURCE) $(ARGS)

run: ## Associate then compliance end-to-end (NAME=, SOURCE=<path>, WEIGHTS=<path> overrides, ARGS="--min-containment 0.6" applies to associate only)
	@$(PYTHON) pipeline/associate.py --weights $(if $(WEIGHTS),$(WEIGHTS),runs/train/$(NAME)/weights/best.pt) --source $(SOURCE) --output runs/associate/$(NAME) $(ARGS)
	@$(PYTHON) pipeline/compliance.py --associations runs/associate/$(NAME)/associations.jsonl --output runs/compliance/$(NAME)
	@echo ""
	@echo "associate output: runs/associate/$(NAME)/ (associations.jsonl, annotated video, associate_summary.json)"
	@echo "compliance output: runs/compliance/$(NAME)/ (events.jsonl, compliance_summary.json)"

serve: ## Serve docs live at http://localhost:8000
	@$(COMPOSE) up -d

docs: ## Build the static docs site into site/
	@$(COMPOSE) run --rm docs build

clean: ## Remove generated runs and the built docs site
	@rm -rf site/
