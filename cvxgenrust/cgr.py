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
    output_dir = Path(code_dir)
    return CodeGenerator(
        module_name=module_name,
        wrapper=wrapper,
        verbose=verbose,
    ).generate(
        problem,
        code_dir=output_dir,
    )
