"""Public Python API for cvxgenrust.

Use :func:`generate_code` to turn a parameterized CVXPY problem into a
generated Rust solver project.
"""

from . import cgr
from .specs import GeneratedRustProject

generate_code = cgr.generate_code

__all__ = ["GeneratedRustProject", "generate_code", "cgr"]
