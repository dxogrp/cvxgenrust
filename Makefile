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

.PHONY: build
build: sync ## build source and wheel distributions
	@printf "$(BLUE)Building package artifacts...$(RESET)\n"
	@rm -rf dist
	@uv build

.PHONY: release-check
release-check: sync ## run release validation checks
	@printf "$(BLUE)Running release test suite...$(RESET)\n"
	@CARGO_TARGET_DIR=$${CARGO_TARGET_DIR:-$(CURDIR)/target/cvxgenrust-ci} uv run python -m pytest tests -m "not (sdp and (numerical or rust_smoke))"
	@printf "$(BLUE)Building release artifacts...$(RESET)\n"
	@rm -rf dist
	@uv build --no-sources
	@printf "$(BLUE)Checking release artifacts...$(RESET)\n"
	@PACKAGE_VERSION="$$(uv run --frozen python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"; \
		uv run --frozen --group dev python scripts/verify_release.py --version "$$PACKAGE_VERSION"
	@printf "$(BLUE)Smoke-testing release artifacts...$(RESET)\n"
	@PACKAGE_VERSION="$$(uv run --frozen python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"; \
		for distribution in dist/*.whl dist/*.tar.gz; do \
			CVXGENRUST_RELEASE_VERSION="$$PACKAGE_VERSION" \
			CVXGENRUST_REPOSITORY_ROOT="$(CURDIR)" \
			uv run --isolated --no-project --with "$$distribution" -- python scripts/release_smoke.py \
				|| exit 1; \
		done

.PHONY: marimo
marimo: sync-examples ## open Marimo apps from the examples directory
	@printf "$(BLUE)Opening Marimo examples...$(RESET)\n"
	@cd examples && uv run --group examples marimo edit .

.PHONY: clean
clean: ## remove local build and test artifacts
	@printf "$(BLUE)Cleaning local artifacts...$(RESET)\n"
	@rm -rf .pytest_cache build dist *.egg-info src/*.egg-info

.PHONY: help
help: ## display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
