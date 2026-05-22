from __future__ import annotations

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


def _parameter_pack_kind(
    internal_parameter: cp.Parameter,
    original_parameters_by_name: dict[str, cp.Parameter],
) -> str | None:
    original_parameter = original_parameters_by_name.get(internal_parameter.name() or "")
    if original_parameter is None:
        return None

    original_shape = tuple(int(x) for x in original_parameter.shape)
    if len(original_shape) != 2 or original_shape[0] != original_shape[1]:
        return None

    n = original_shape[0]
    if (
        int(internal_parameter.size) == n * (n + 1) // 2
        and int(original_parameter.size) == n * n
    ):
        return "upper_tri"
    return None


def extract_problem(
    problem: cp.Problem,
    module_name: str,
) -> ProblemSpec:
    if not problem.is_dpp(quad_form_dpp="qp"):
        raise ValueError("problem must satisfy CVXPY's DPP rules for code generation")
    cvxpy_solver = cp.CLARABEL
    data, _, inverse_data = problem.get_problem_data(cvxpy_solver)
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
    original_parameters_by_name = {
        parameter.name(): parameter
        for parameter in problem.parameters()
        if parameter.name() is not None
    }
    for parameter in param_prob.parameters:
        name = parameter.name() or f"param_{parameter.id}"
        offset = int(param_prob.param_id_to_col[parameter.id])
        parameters.append(
            ParameterSpec(
                name=name,
                shape=tuple(int(x) for x in parameter.shape),
                size=int(parameter.size),
                offset=offset,
                pack=_parameter_pack_kind(parameter, original_parameters_by_name),
            )
        )

    variables = []
    for variable in problem.variables():
        if variable.id not in param_prob.var_id_to_col:
            continue
        name = variable.name() or f"var_{variable.id}"
        offset = int(param_prob.var_id_to_col[variable.id])
        variables.append(
            VariableSpec(
                name=name,
                shape=tuple(int(x) for x in variable.shape),
                size=int(variable.size),
                offset=offset,
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
