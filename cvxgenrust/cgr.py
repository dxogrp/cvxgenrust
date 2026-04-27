from pathlib import Path

import cvxpy as cp

from .generator import GeneratedRustProject, generate_code as _generate_code


def generate_code(
    problem: cp.Problem,
    code_dir: str | Path = "CGR_code",
    module_name: str | None = None,
) -> GeneratedRustProject:
    output_dir = Path(code_dir)
    full_module_name = module_name or output_dir.name
    return _generate_code(
        problem,
        code_dir=output_dir,
        module_name=full_module_name,
    )
