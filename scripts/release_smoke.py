"""Smoke-test an installed cvxgenrust release distribution."""

from __future__ import annotations

import importlib.metadata
import os
import tempfile
from pathlib import Path

import cvxpy as cp
import numpy as np

import cvxgenrust as cgr
from cvxgenrust.config import GENERATOR_VERSION


def _smoke_generate_code() -> None:
    a = cp.Parameter((2, 1), name="A")
    b = cp.Parameter(2, name="b")
    x = cp.Variable(1, name="x")
    a.value = np.array([[1.0], [2.0]])
    b.value = np.array([1.0, 2.0])
    problem = cp.Problem(cp.Minimize(cp.sum_squares(a @ x - b)), [x >= 0])

    with tempfile.TemporaryDirectory(prefix="cvxgenrust-release-smoke-") as tmpdir:
        workspace = Path(tmpdir)
        output_dir = workspace / "tiny_solver_cgr"
        project = cgr.generate_code(
            problem,
            code_dir=output_dir,
            module_name="tiny_solver",
            wrapper=False,
            verbose=False,
        )

        if project.spec.module_name != "tiny_solver" or project.output_dir.resolve() != output_dir.resolve():
            raise RuntimeError("Release smoke generation returned inconsistent project metadata.")
        expected_files = {
            "Cargo.toml",
            "LICENSE",
            "README.html",
            "examples/solve.rs",
            "pyproject.toml",
            "python/tiny_solver_wrapper/__init__.py",
            "python/tiny_solver_wrapper/cgr_solver.py",
            "src/data.rs",
            "src/lib.rs",
            "src/runtime.rs",
        }
        missing = sorted(
            relative
            for relative in expected_files
            if not (output_dir / relative).is_file() or not (output_dir / relative).read_bytes()
        )
        if missing:
            raise RuntimeError(f"Release smoke generation is missing files: {', '.join(missing)}.")


def main() -> int:
    expected_version = os.environ["CVXGENRUST_RELEASE_VERSION"]
    repository_root = Path(os.environ["CVXGENRUST_REPOSITORY_ROOT"]).resolve()
    installed_package = Path(cgr.__file__).resolve()
    if installed_package.is_relative_to(repository_root):
        raise RuntimeError(
            f"Smoke test imported the repository instead of the isolated distribution: {installed_package}."
        )

    distribution_version = importlib.metadata.version("cvxgenrust")
    if distribution_version != expected_version:
        raise RuntimeError(
            f"Installed cvxgenrust version is {distribution_version!r}, expected {expected_version!r}."
        )
    if GENERATOR_VERSION != expected_version:
        raise RuntimeError(
            f"Installed generator version is {GENERATOR_VERSION!r}, expected {expected_version!r}."
        )

    _smoke_generate_code()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
