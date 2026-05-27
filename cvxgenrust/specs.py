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


@dataclass
class ParameterSpec:
    name: str
    shape: tuple[int, ...]
    size: int
    offset: int
    pack: str | None = None


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
    spec: ProblemSpec
    output_dir: Path
