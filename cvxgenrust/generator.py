from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cvxpy as cp

from .config import (
    CLARABEL_VERSION,
    GENERATED_PYTHON_DEPENDENCIES,
    GENERATED_REQUIRES_PYTHON,
    GENERATOR_VERSION,
)
from .extract import extract_problem
from .names import _python_package_name, _snake_case
from .render import (
    _render_generated_cargo,
    _render_generated_init,
    _render_generated_lib,
    _render_generated_license,
    _render_generated_pyproject,
    _render_generated_python_wrapper,
    _render_generated_readme,
    _render_generated_rust_example,
)
from .specs import GeneratedRustProject


def _python_install_command(target_dir: Path, project_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--target",
        str(target_dir),
        "--upgrade",
        str(project_dir),
    ]


def _ensure_pip_available() -> None:
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if pip_check.returncode == 0:
        return

    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "automatic wrapper compilation requires pip; install pip or run with wrapper=False"
        ) from error


def _compile_python_wrapper(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    python_source_dir = output_dir / "python"
    python_source_dir.mkdir(parents=True, exist_ok=True)
    _ensure_pip_available()
    subprocess.run(
        _python_install_command(python_source_dir, output_dir),
        cwd=output_dir,
        check=True,
    )


def generate_code(
    problem: cp.Problem,
    code_dir: str | Path,
    module_name: str,
    wrapper: bool = True,
    verbose: bool = True,
) -> GeneratedRustProject:
    output_dir = Path(code_dir)
    module = module_name
    package_name = _python_package_name(f"{module_name}_wrapper")

    def progress(message: str) -> None:
        if verbose:
            print(f"[CvxGenRust] {message}", file=sys.stderr)

    progress(f"Extracting problem data for module {module!r}")
    spec = extract_problem(problem, module_name=module)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    src_dir = output_dir / "src"
    package_dir = output_dir / "python" / package_name
    progress("Creating output directories")
    src_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)

    progress("Writing Rust crate and Python package files")
    (output_dir / "Cargo.toml").write_text(
        _render_generated_cargo(spec, generated_at), encoding="utf-8"
    )
    (output_dir / "pyproject.toml").write_text(
        _render_generated_pyproject(spec, package_name, generated_at), encoding="utf-8"
    )
    (output_dir / "LICENSE").write_text(
        _render_generated_license(spec, generated_at), encoding="utf-8"
    )
    (package_dir / "__init__.py").write_text(
        _render_generated_init(spec, generated_at), encoding="utf-8"
    )
    (src_dir / "lib.rs").write_text(_render_generated_lib(spec, generated_at), encoding="utf-8")
    (output_dir / "README.html").write_text(
        _render_generated_readme(spec, package_name, output_dir.name, generated_at), encoding="utf-8"
    )
    (package_dir / "cgr_solver.py").write_text(
        _render_generated_python_wrapper(spec, generated_at), encoding="utf-8"
    )
    examples_dir = output_dir / "examples"
    progress("Writing Rust example")
    examples_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "solve.rs").write_text(
        _render_generated_rust_example(spec, generated_at), encoding="utf-8"
    )

    if wrapper:
        progress("Compiling Python extension wrapper")
        _compile_python_wrapper(output_dir)

    progress(f"Done. Generated solver project at {output_dir}")
    return GeneratedRustProject(
        spec=spec,
        output_dir=output_dir,
    )


__all__ = [
    "CLARABEL_VERSION",
    "GENERATED_PYTHON_DEPENDENCIES",
    "GENERATED_REQUIRES_PYTHON",
    "GENERATOR_VERSION",
    "GeneratedRustProject",
    "_python_package_name",
    "_snake_case",
    "extract_problem",
    "generate_code",
]
