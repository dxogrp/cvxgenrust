from __future__ import annotations

import keyword
import re

_RUST_KEYWORDS = {
    "as",
    "async",
    "await",
    "box",
    "break",
    "const",
    "continue",
    "crate",
    "dyn",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "union",
    "unsafe",
    "use",
    "where",
    "while",
    "abstract",
    "become",
    "do",
    "final",
    "macro",
    "override",
    "priv",
    "try",
    "typeof",
    "unsized",
    "virtual",
    "yield",
}


def _snake_case(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower() or "cgr_solver"


def _rust_module_name(name: str) -> str:
    candidate = _snake_case(name)
    if candidate[0].isdigit():
        candidate = f"cgr_{candidate}"
    if candidate in _RUST_KEYWORDS or keyword.iskeyword(candidate):
        candidate = f"{candidate}_solver"
    return candidate


def _wrapper_package_name(module_name: str) -> str:
    return f"{_rust_module_name(module_name)}_wrapper"


def _python_distribution_name(package_name: str) -> str:
    return package_name.replace("_", "-")


def _rust_ident(name: str) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    if not candidate:
        candidate = "unnamed_value"
    if candidate[0].isdigit():
        candidate = f"p_{candidate}"
    if candidate in _RUST_KEYWORDS or keyword.iskeyword(candidate):
        candidate = f"{candidate}_value"
    return candidate
