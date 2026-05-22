from . import cgr
from .specs import GeneratedRustProject

generate_code = cgr.generate_code

__all__ = ["GeneratedRustProject", "generate_code", "cgr"]
