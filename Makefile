BLUE := \033[36m
BOLD := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: sync
sync: ## install the default development environment
	@printf "$(BLUE)Syncing development dependencies...$(RESET)\n"
	@uv sync --frozen --group dev

.PHONY: sync-examples
sync-examples:
	@printf "$(BLUE)Syncing example dependencies...$(RESET)\n"
	@uv sync --frozen --group dev --group examples

.PHONY: test
test: sync ## run the test suite
	@printf "$(BLUE)Running tests...$(RESET)\n"
	@uv run pytest tests

.PHONY: marimo
marimo: sync-examples ## open Marimo apps from the examples directory
	@printf "$(BLUE)Opening Marimo examples...$(RESET)\n"
	@uv run --group examples marimo edit examples

.PHONY: clean
clean: ## remove local build and test artifacts
	@printf "$(BLUE)Cleaning local artifacts...$(RESET)\n"
	@rm -rf .pytest_cache

.PHONY: help
help: ## display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
