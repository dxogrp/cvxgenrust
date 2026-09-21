from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MatrixPatternSpec:
    rows: int
    cols: int
    indices: list[int]
    indptr: list[int]


@dataclass
class CsrMatrixSpec:
    rows: int
    cols: int
    indptr: list[int]
    indices: list[int]
    data: list[float]


@dataclass
class AffineCscMapSpec:
    reduced: CsrMatrixSpec
    pattern: MatrixPatternSpec


@dataclass
class AffineVectorMapSpec:
    reduced: CsrMatrixSpec
    output_len: int


@dataclass(frozen=True)
class ParameterLayoutSpec:
    kind: str
    flat_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        supported_kinds = {
            "dense_column_major",
            "diagonal",
            "symmetric_upper_triangle",
            "sparse_diagonal",
            "sparse_upper_triangle",
            "sparse_lower_triangle",
            "sparse_explicit",
        }
        if self.kind not in supported_kinds:
            raise ValueError(f"unsupported parameter layout kind: {self.kind!r}")
        if self.kind != "sparse_explicit" and self.flat_indices:
            raise ValueError(
                "flat parameter indices are only valid for sparse_explicit layouts"
            )


@dataclass
class ParameterSpec:
    name: str
    shape: tuple[int, ...]
    size: int
    offset: int
    layout: ParameterLayoutSpec


@dataclass
class VariableSpec:
    name: str
    shape: tuple[int, ...]
    size: int
    canonical_size: int
    offset: int
    unpack: str | None = None


@dataclass
class DualVariableSpec:
    name: str
    shape: tuple[int, ...]
    size: int
    offset: int


@dataclass
class ConeDimsSpec:
    zero: int
    nonneg: int
    exp: int
    soc: list[int]
    psd: list[int]
    p3d: list[float]


@dataclass
class ProblemSpec:
    module_name: str
    parameter_vec_len: int
    cone_dims: ConeDimsSpec
    parameters: list[ParameterSpec]
    variables: list[VariableSpec]
    dual_variables: list[DualVariableSpec]
    p_map: AffineCscMapSpec
    a_map: AffineCscMapSpec
    q_map: AffineVectorMapSpec


@dataclass
class GeneratedRustProject:
    """Description of a generated Rust solver project.

    Attributes
    ----------
    spec:
        Extracted problem metadata used to render the solver.
    output_dir:
        Directory containing the generated Cargo project and Python wrapper
        sources.
    """

    spec: ProblemSpec
    output_dir: Path
