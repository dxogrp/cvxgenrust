from pathlib import Path

import cvxpy as cp

from .generator import GeneratedRustProject, generate_code as _generate_code


def generate_code(
    problem: cp.Problem,
    code_dir: str | Path = "cgr_code",
    module_name: str = "cgr_module",
    wrapper: bool = True,
    verbose: bool = True,
) -> GeneratedRustProject:
    output_dir = Path(code_dir)
    return _generate_code(
        problem,
        code_dir=output_dir,
        module_name=module_name,
        wrapper=wrapper,
        verbose=verbose,
    )
