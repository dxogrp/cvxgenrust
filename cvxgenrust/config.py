from __future__ import annotations

import importlib.metadata
from pathlib import Path

TEMPLATES_DIR = Path(__file__).with_name("templates")
GENERATOR_DISPLAY_NAME = "cvxgenrust"


def _load_generator_version() -> str:
    try:
        return importlib.metadata.version("cvxgenrust")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0.dev0"


GENERATOR_VERSION = _load_generator_version()
CLARABEL_VERSION = "0.11.1"
GENERATED_REQUIRES_PYTHON = ">=3.12"
MATURIN_VERSION = ">=1.7,<2"
PYO3_VERSION = "0.25"
GENERATED_PYTHON_DEPENDENCIES = ["cvxpy>=1.9", "numpy>=1.26"]
