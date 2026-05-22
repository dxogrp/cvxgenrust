from __future__ import annotations

import keyword
import re


def _snake_case(name: str) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower() or "cgr_solver"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_solver"
    return candidate


def _python_package_name(name: str) -> str:
    candidate = _snake_case(name)
    if candidate[0].isdigit():
        candidate = f"cgr_{candidate}"
    return candidate


def _python_project_name(package_name: str) -> str:
    return package_name.replace("_", "-")


def _rust_ident(name: str) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    if not candidate:
        candidate = "generated"
    if candidate[0].isdigit():
        candidate = f"p_{candidate}"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_value"
    return candidate
