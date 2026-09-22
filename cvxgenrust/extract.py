from __future__ import annotations

import warnings
from typing import Any

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .names import _rust_module_name
from .specs import (
    AffineCscMapSpec,
    AffineVectorMapSpec,
    ConeDimsSpec,
    CsrMatrixSpec,
    DualVariableSpec,
    MatrixPatternSpec,
    ParameterLayoutSpec,
    ParameterSpec,
    ProblemSpec,
    VariableSpec,
)


def _csr_spec(matrix: sp.spmatrix | sp.sparray) -> CsrMatrixSpec:
    csr = sp.csr_array(matrix)
    return CsrMatrixSpec(
        rows=int(csr.shape[0]),
        cols=int(csr.shape[1]),
        indptr=[int(x) for x in csr.indptr.tolist()],
        indices=[int(x) for x in csr.indices.tolist()],
        data=[float(x) for x in csr.data.tolist()],
    )


def _pattern_spec(indices: Any, indptr: Any, shape: tuple[int, int]) -> MatrixPatternSpec:
    return MatrixPatternSpec(
        rows=int(shape[0]),
        cols=int(shape[1]),
        indices=[int(x) for x in np.asarray(indices).tolist()],
        indptr=[int(x) for x in np.asarray(indptr).tolist()],
    )


def _extract_map(reduced_mat_obj: Any) -> AffineCscMapSpec:
    reduced_mat_obj.cache(True)
    if reduced_mat_obj.problem_data_index is None or reduced_mat_obj.reduced_mat is None:
        raise ValueError("reduced matrix does not expose a problem-data sparsity pattern")
    indices, indptr, shape = reduced_mat_obj.problem_data_index
    return AffineCscMapSpec(
        reduced=_csr_spec(reduced_mat_obj.reduced_mat),
        pattern=_pattern_spec(indices, indptr, shape),
    )


def _extract_vector_map(q_map: sp.spmatrix | sp.sparray) -> AffineVectorMapSpec:
    csr = _csr_spec(q_map)
    return AffineVectorMapSpec(reduced=csr, output_len=csr.rows - 1)


def _extract_cone_dims(dims: Any) -> ConeDimsSpec:
    return ConeDimsSpec(
        zero=int(getattr(dims, "zero", 0)),
        nonneg=int(getattr(dims, "nonneg", 0)),
        exp=int(getattr(dims, "exp", 0)),
        soc=[int(x) for x in getattr(dims, "soc", [])],
        psd=[int(x) for x in getattr(dims, "psd", [])],
        p3d=[float(x) for x in getattr(dims, "p3d", [])],
    )


def _zero_csc_map_spec(rows: int, cols: int, parameter_vec_len: int) -> AffineCscMapSpec:
    return AffineCscMapSpec(
        reduced=CsrMatrixSpec(
            rows=0,
            cols=parameter_vec_len,
            indptr=[0],
            indices=[],
            data=[],
        ),
        pattern=MatrixPatternSpec(
            rows=rows,
            cols=cols,
            indices=[],
            indptr=[0] * (cols + 1),
        ),
    )


def _upper_triangle_size(rows: int, cols: int) -> int:
    overlap = min(rows, cols)
    return overlap * (overlap + 1) // 2 + max(cols - rows, 0) * rows


def _lower_triangle_size(rows: int, cols: int) -> int:
    overlap = min(rows, cols)
    return overlap * (overlap + 1) // 2 + max(rows - cols, 0) * cols


def _coordinates_are_row_major(rows: np.ndarray, cols: np.ndarray) -> bool:
    if rows.size < 2:
        return True
    return bool(
        np.all(
            (rows[1:] > rows[:-1])
            | ((rows[1:] == rows[:-1]) & (cols[1:] > cols[:-1]))
        )
    )


def _sparse_parameter_layout(parameter: cp.Parameter) -> ParameterLayoutSpec:
    shape = tuple(int(x) for x in parameter.shape)
    sparse_idx = getattr(parameter, "sparse_idx", None)
    if sparse_idx is None:
        raise ValueError(
            f"parameter {parameter.name() or parameter.id!r} has no canonical sparsity indices"
        )

    coordinates = tuple(
        np.asarray(axis, dtype=np.int64).reshape(-1) for axis in sparse_idx
    )
    if len(coordinates) != len(shape):
        raise ValueError(
            f"parameter {parameter.name() or parameter.id!r} has a sparsity pattern "
            "whose rank does not match its shape"
        )
    coordinate_lengths = {int(axis.size) for axis in coordinates}
    if len(coordinate_lengths) > 1:
        raise ValueError(
            f"parameter {parameter.name() or parameter.id!r} has inconsistent sparsity "
            "coordinate lengths"
        )

    if len(shape) == 2:
        row_count, col_count = shape
        row_indices, col_indices = coordinates
        entry_count = int(row_indices.size)
        row_major = _coordinates_are_row_major(row_indices, col_indices)
        if (
            entry_count == min(row_count, col_count)
            and row_major
            and bool(np.all(row_indices == col_indices))
        ):
            return ParameterLayoutSpec(kind="sparse_diagonal")
        if (
            entry_count == _upper_triangle_size(row_count, col_count)
            and row_major
            and bool(np.all(col_indices >= row_indices))
        ):
            return ParameterLayoutSpec(kind="sparse_upper_triangle")
        if (
            entry_count == _lower_triangle_size(row_count, col_count)
            and row_major
            and bool(np.all(col_indices <= row_indices))
        ):
            return ParameterLayoutSpec(kind="sparse_lower_triangle")

    flat_indices = tuple(
        int(index)
        for index in np.ravel_multi_index(coordinates, shape, order="F").tolist()
    )
    return ParameterLayoutSpec(kind="sparse_explicit", flat_indices=flat_indices)


def _parameter_layout(parameter: cp.Parameter) -> ParameterLayoutSpec:
    attributes = getattr(parameter, "attributes", {})
    if parameter.is_complex() or attributes.get("hermitian", False):
        raise ValueError(
            f"parameter {parameter.name() or parameter.id!r} uses a complex or Hermitian "
            "layout, which cvxgenrust does not support"
        )

    shape = tuple(int(x) for x in parameter.shape)
    if attributes.get("diag", False):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                f"diagonal parameter {parameter.name() or parameter.id!r} must be a square matrix"
            )
        return ParameterLayoutSpec(kind="diagonal")

    if any(attributes.get(name, False) for name in ("symmetric", "PSD", "NSD")):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                f"symmetric parameter {parameter.name() or parameter.id!r} must be a square matrix"
            )
        return ParameterLayoutSpec(kind="symmetric_upper_triangle")

    if getattr(parameter, "sparse_idx", None) is not None:
        return _sparse_parameter_layout(parameter)

    return ParameterLayoutSpec(kind="dense_column_major")


def _expected_parameter_size(
    parameter: cp.Parameter,
    layout: ParameterLayoutSpec,
) -> int:
    shape = tuple(int(x) for x in parameter.shape)
    if layout.kind == "dense_column_major":
        return int(parameter.size)
    if layout.kind in {"diagonal", "sparse_diagonal"}:
        return min(shape)
    if layout.kind in {"symmetric_upper_triangle", "sparse_upper_triangle"}:
        rows, cols = shape
        return _upper_triangle_size(rows, cols)
    if layout.kind == "sparse_lower_triangle":
        rows, cols = shape
        return _lower_triangle_size(rows, cols)
    if layout.kind == "sparse_explicit":
        return len(layout.flat_indices)
    raise ValueError(f"unsupported parameter layout kind: {layout.kind!r}")


def _canonical_parameter(
    original_parameter: cp.Parameter,
    composed_id_map: dict[int, list[int]],
    canonical_parameters_by_id: dict[int, cp.Parameter],
) -> cp.Parameter:
    canonical_ids = composed_id_map.get(original_parameter.id, [original_parameter.id])
    if len(canonical_ids) != 1:
        raise ValueError(
            f"parameter {original_parameter.name() or original_parameter.id!r} maps to "
            f"{len(canonical_ids)} canonical parameter blocks {canonical_ids}; "
            "cvxgenrust requires exactly one block per parameter"
        )
    canonical_id = int(canonical_ids[0])
    canonical_parameter = canonical_parameters_by_id.get(canonical_id)
    if canonical_parameter is None:
        raise ValueError(
            f"parameter {original_parameter.name() or original_parameter.id!r} maps to "
            f"canonical parameter ID {canonical_id}, but that block is unavailable"
        )
    return canonical_parameter


def _variable_unpack_kind(variable: cp.Variable) -> str | None:
    shape = tuple(int(x) for x in variable.shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        return None

    attributes = getattr(variable, "attributes", {})
    if any(attributes.get(name, False) for name in ("symmetric", "PSD", "NSD")):
        return "symmetric"
    return None


def _canonical_variable_sizes(
    inverse_data: list[Any],
    canonical_dim: int,
) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for item in inverse_data:
        item_dict = getattr(item, "__dict__", None)
        if not item_dict or item_dict.get("x_length") != canonical_dim:
            continue
        id_map = item_dict.get("id_map", {})
        sizes.update({int(var_id): int(size) for var_id, (_offset, size) in id_map.items()})
    return sizes


def extract_problem(
    problem: cp.Problem,
    module_name: str,
) -> ProblemSpec:
    if not problem.is_dpp(quad_form_dpp="qp"):
        raise ValueError("problem must satisfy CVXPY's DPP rules for code generation")
    cvxpy_solver = cp.CLARABEL
    # CVXPY 1.9 reads sparse parameters through `.value` internally while
    # constructing the reduction chain, even when callers correctly use
    # `.value_sparse`. Suppress only that known, spurious warning here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Reading from a sparse CVXPY expression via `\.value` is discouraged\..*",
            category=RuntimeWarning,
        )
        data, chain, inverse_data = problem.get_problem_data(cvxpy_solver)
    param_prob = data["param_prob"]
    parameter_vec_len = int(param_prob.total_param_size + 1)
    canonical_dim = int(getattr(param_prob.x, "size", data["A"].shape[1]))
    if param_prob.reduced_P.problem_data_index is None or param_prob.reduced_P.reduced_mat is None:
        p_map = _zero_csc_map_spec(canonical_dim, canonical_dim, parameter_vec_len)
    else:
        p_map = _extract_map(param_prob.reduced_P)
    a_map = _extract_map(param_prob.reduced_A)
    linear_obj_tensor = getattr(param_prob, "q", None)
    if linear_obj_tensor is None:
        linear_obj_tensor = getattr(param_prob, "c")
    linear_obj_map = _extract_vector_map(linear_obj_tensor)
    dims = _extract_cone_dims(data["dims"])
    parameters = []
    composed_param_id_map = chain.compose_param_id_map()
    canonical_parameters_by_id = {
        int(parameter_id): parameter
        for parameter_id, parameter in param_prob.id_to_param.items()
    }
    for original_parameter in problem.parameters():
        layout = _parameter_layout(original_parameter)
        canonical_parameter = _canonical_parameter(
            original_parameter,
            composed_param_id_map,
            canonical_parameters_by_id,
        )
        packed_size = int(canonical_parameter.size)
        expected_size = _expected_parameter_size(original_parameter, layout)
        if packed_size != expected_size:
            name = original_parameter.name() or f"param_{original_parameter.id}"
            raise ValueError(
                f"parameter {name!r} has {layout.kind!r} layout with expected packed "
                f"size {expected_size}, but its canonical block has size {packed_size}"
            )
        canonical_id = int(canonical_parameter.id)
        if canonical_id not in param_prob.param_id_to_col:
            name = original_parameter.name() or f"param_{original_parameter.id}"
            raise ValueError(
                f"parameter {name!r} maps to canonical parameter ID {canonical_id}, "
                "but that block has no parameter-vector offset"
            )
        name = original_parameter.name() or f"param_{original_parameter.id}"
        offset = int(param_prob.param_id_to_col[canonical_id])
        parameters.append(
            ParameterSpec(
                name=name,
                shape=tuple(int(x) for x in original_parameter.shape),
                size=packed_size,
                offset=offset,
                layout=layout,
            )
        )

    variables = []
    canonical_variable_sizes = _canonical_variable_sizes(inverse_data, canonical_dim)
    for variable in problem.variables():
        if variable.id not in param_prob.var_id_to_col:
            continue
        name = variable.name() or f"var_{variable.id}"
        offset = int(param_prob.var_id_to_col[variable.id])
        canonical_size = canonical_variable_sizes.get(variable.id, int(variable.size))
        variables.append(
            VariableSpec(
                name=name,
                shape=tuple(int(x) for x in variable.shape),
                size=int(variable.size),
                canonical_size=canonical_size,
                offset=offset,
                unpack=_variable_unpack_kind(variable),
            )
        )

    solver_inverse = inverse_data[-1].inverse_data if inverse_data else {}
    canonical_constraints = list(solver_inverse.get("eq_constr", [])) + list(
        solver_inverse.get("other_constr", [])
    )
    dual_variables = []
    dual_offset = 0
    for index, constraint in enumerate(canonical_constraints):
        size = int(constraint.size)
        dual_variables.append(
            DualVariableSpec(
                name=f"d{index}",
                shape=tuple(int(x) for x in constraint.shape),
                size=size,
                offset=dual_offset,
            )
        )
        dual_offset += size

    return ProblemSpec(
        module_name=_rust_module_name(module_name),
        parameter_vec_len=parameter_vec_len,
        cone_dims=dims,
        parameters=parameters,
        variables=variables,
        dual_variables=dual_variables,
        p_map=p_map,
        a_map=a_map,
        q_map=linear_obj_map,
    )
