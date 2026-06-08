from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

TEMPLATES_DIR = Path(__file__).with_name("templates")
PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"
GENERATOR_DISPLAY_NAME = "cvxgenrust"


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _load_generator_version() -> str:
    project_data = _load_pyproject()
    return str(project_data["project"]["version"])


def _load_clarabel_version() -> str:
    project_data = _load_pyproject()
    return str(project_data.get("tool", {}).get("cvxgenrust", {}).get("clarabel-version", "0.11.1"))


def _load_generated_python_dependencies() -> list[str]:
    project_data = _load_pyproject()
    dependencies = project_data.get("tool", {}).get("cvxgenrust", {}).get(
        "python-dependencies",
        ["cvxpy>=1.9", "numpy>=1.26"],
    )
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        raise TypeError("[tool.cvxgenrust].python-dependencies must be a list of strings")
    return dependencies


def _load_tool_config_string(name: str, default: str) -> str:
    project_data = _load_pyproject()
    return str(project_data.get("tool", {}).get("cvxgenrust", {}).get(name, default))


GENERATOR_VERSION = _load_generator_version()
CLARABEL_VERSION = _load_clarabel_version()
GENERATED_REQUIRES_PYTHON = _load_tool_config_string("generated-requires-python", ">=3.12")
MATURIN_VERSION = _load_tool_config_string("maturin-version", ">=1.7,<2")
PYO3_VERSION = _load_tool_config_string("pyo3-version", "0.25")
GENERATED_PYTHON_DEPENDENCIES = _load_generated_python_dependencies()
