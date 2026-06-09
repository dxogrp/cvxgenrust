from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path
from typing import Any

TEMPLATES_DIR = Path(__file__).with_name("templates")
PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"
GENERATOR_DISPLAY_NAME = "cvxgenrust"


def _load_pyproject() -> dict[str, Any]:
    if not PYPROJECT_PATH.exists():
        return {}
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _load_generator_version() -> str:
    project_data = _load_pyproject()
    project_version = project_data.get("project", {}).get("version")
    if project_version is not None:
        return str(project_version)
    try:
        return importlib.metadata.version("cvxgenrust")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


GENERATOR_VERSION = _load_generator_version()
CLARABEL_VERSION = "0.11.1"
GENERATED_REQUIRES_PYTHON = ">=3.12"
MATURIN_VERSION = ">=1.7,<2"
PYO3_VERSION = "0.25"
GENERATED_PYTHON_DEPENDENCIES = ["cvxpy>=1.9", "numpy>=1.26"]
