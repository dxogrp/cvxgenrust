from pathlib import Path

import cvxpy as cp

from .generator import CodeGenerator
from .specs import GeneratedRustProject


def generate_code(
    problem: cp.Problem,
    code_dir: str | Path = "cgr_code",
    module_name: str = "cgr_module",
    wrapper: bool = True,
    verbose: bool = True,
) -> GeneratedRustProject:
    """Generate a Rust solver project from a parameterized CVXPY problem.

    Parameters
    ----------
    problem:
        CVXPY problem to canonicalize and render. Parameters and variables
        should be created with ``name=`` so the generated setters and extractors
        have stable names.
    code_dir:
        Output directory for the generated Cargo project, generated Python
        package sources, examples, license, and ``README.html``.
    module_name:
        Rust crate/module name for the generated solver. The Python wrapper
        package is derived from this name.
    wrapper:
        Whether to compile the generated PyO3 extension wrapper into
        ``code_dir/python`` after writing the project files.
    verbose:
        Whether to print generation progress to stderr.

    Returns
    -------
    GeneratedRustProject
        Metadata for the generated solver and the output directory.
    """

    output_dir = Path(code_dir)
    return CodeGenerator(
        module_name=module_name,
        wrapper=wrapper,
        verbose=verbose,
    ).generate(
        problem,
        code_dir=output_dir,
    )
