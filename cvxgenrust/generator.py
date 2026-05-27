from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
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
from .names import _rust_ident, _snake_case, _wrapper_package_name
from .render import (
    _render_generated_cargo,
    _render_generated_data,
    _render_generated_init,
    _render_generated_lib,
    _render_generated_license,
    _render_generated_pyproject,
    _render_generated_python_wrapper,
    _render_generated_readme,
    _render_generated_runtime,
    _render_generated_rust_example,
)
from .specs import GeneratedRustProject, ProblemSpec

_RESERVED_PARAMETER_METHOD_IDENTS = {
    "parameter",
    "parameter_entry",
    "solver_default_settings",
    "solver_verbose",
    "solver_max_iter",
    "solver_time_limit",
    "solver_tol_gap_abs",
    "solver_tol_gap_rel",
    "solver_tol_feas",
}
_RESERVED_EXTRACTOR_METHOD_IDENTS = {"variable", "dual_variable"}


def _rust_ident_clashes(
    kind: str,
    names: list[str],
    reserved: set[str],
) -> list[str]:
    seen: dict[str, str] = {}
    clashes = []
    for name in names:
        ident = _rust_ident(name)
        if ident in reserved:
            clashes.append(f"{kind} {name!r} maps to reserved Rust API name {ident!r}")
            continue
        if ident in seen:
            clashes.append(
                f"{kind} names {seen[ident]!r} and {name!r} both map to Rust API name {ident!r}"
            )
            continue
        seen[ident] = name
    return clashes


def _validate_rust_api_names(spec: ProblemSpec) -> None:
    clashes = [
        *_rust_ident_clashes(
            "parameter",
            [parameter.name for parameter in spec.parameters],
            _RESERVED_PARAMETER_METHOD_IDENTS,
        ),
        *_rust_ident_clashes(
            "variable/dual variable",
            [variable.name for variable in spec.variables]
            + [dual_variable.name for dual_variable in spec.dual_variables],
            _RESERVED_EXTRACTOR_METHOD_IDENTS,
        ),
    ]
    if clashes:
        detail = "; ".join(clashes)
        raise ValueError(
            "generated Rust API method name clash; rename the CVXPY parameters, "
            f"variables, or constraints. {detail}"
        )


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


@dataclass
class CodeGenerator:
    module_name: str
    wrapper: bool = True
    verbose: bool = True

    @property
    def package_name(self) -> str:
        return _wrapper_package_name(self.module_name)

    def generate(
        self,
        problem: cp.Problem,
        code_dir: str | Path,
    ) -> GeneratedRustProject:
        output_dir = Path(code_dir)
        self.progress(f"Extracting problem data for module {self.module_name!r}")
        spec = self.extract_problem(problem)
        self.validate_problem_spec(spec)
        generated_at = self.generated_timestamp()

        self.prepare_output_dirs(output_dir)
        self.write_project_files(spec, output_dir, generated_at)
        self.write_examples(spec, output_dir, generated_at)
        self.compile_wrapper_if_requested(output_dir)

        self.progress(f"Done. Generated solver project at {output_dir}")
        return GeneratedRustProject(
            spec=spec,
            output_dir=output_dir,
        )

    def progress(self, message: str) -> None:
        if self.verbose:
            print(f"[CvxGenRust] {message}", file=sys.stderr)

    def generated_timestamp(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def extract_problem(self, problem: cp.Problem) -> ProblemSpec:
        return extract_problem(problem, module_name=self.module_name)

    def validate_problem_spec(self, spec: ProblemSpec) -> None:
        _validate_rust_api_names(spec)

    def prepare_output_dirs(self, output_dir: Path) -> None:
        src_dir = output_dir / "src"
        package_dir = output_dir / "python" / self.package_name
        self.progress("Creating output directories")
        src_dir.mkdir(parents=True, exist_ok=True)
        package_dir.mkdir(parents=True, exist_ok=True)

    def write_project_files(self, spec: ProblemSpec, output_dir: Path, generated_at: str) -> None:
        package_dir = output_dir / "python" / self.package_name
        src_dir = output_dir / "src"
        self.progress("Writing Rust crate and Python package files")
        (output_dir / "Cargo.toml").write_text(
            _render_generated_cargo(spec, generated_at), encoding="utf-8"
        )
        (output_dir / "pyproject.toml").write_text(
            _render_generated_pyproject(spec, self.package_name, generated_at), encoding="utf-8"
        )
        (output_dir / "LICENSE").write_text(
            _render_generated_license(spec, generated_at), encoding="utf-8"
        )
        (package_dir / "__init__.py").write_text(
            _render_generated_init(spec, generated_at), encoding="utf-8"
        )
        (src_dir / "lib.rs").write_text(
            _render_generated_lib(spec, generated_at), encoding="utf-8"
        )
        (src_dir / "runtime.rs").write_text(
            _render_generated_runtime(spec, generated_at), encoding="utf-8"
        )
        (src_dir / "data.rs").write_text(
            _render_generated_data(spec, generated_at), encoding="utf-8"
        )
        (output_dir / "README.html").write_text(
            _render_generated_readme(spec, self.package_name, output_dir.name, generated_at),
            encoding="utf-8",
        )
        (package_dir / "cgr_solver.py").write_text(
            _render_generated_python_wrapper(spec, generated_at), encoding="utf-8"
        )

    def write_examples(self, spec: ProblemSpec, output_dir: Path, generated_at: str) -> None:
        examples_dir = output_dir / "examples"
        self.progress("Writing Rust example")
        examples_dir.mkdir(parents=True, exist_ok=True)
        (examples_dir / "solve.rs").write_text(
            _render_generated_rust_example(spec, generated_at), encoding="utf-8"
        )

    def compile_wrapper_if_requested(self, output_dir: Path) -> None:
        if not self.wrapper:
            return
        self.progress("Compiling Python extension wrapper")
        self.compile_wrapper(output_dir)

    def compile_wrapper(self, output_dir: Path) -> None:
        _compile_python_wrapper(output_dir)


__all__ = [
    "CLARABEL_VERSION",
    "GENERATED_PYTHON_DEPENDENCIES",
    "GENERATED_REQUIRES_PYTHON",
    "GENERATOR_VERSION",
    "CodeGenerator",
    "GeneratedRustProject",
    "_snake_case",
    "extract_problem",
]
