from __future__ import annotations
from datetime import datetime
import html
import keyword
import math
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

TEMPLATES_DIR = Path(__file__).with_name("templates")
PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"
GENERATOR_DISPLAY_NAME = "CvxGenRust"


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _load_generator_version() -> str:
    project_data = _load_pyproject()
    return str(project_data["project"]["version"])


def _load_clarabel_version() -> str:
    project_data = _load_pyproject()
    return str(project_data.get("tool", {}).get("cvxgenrust", {}).get("clarabel-version", "0.11.1"))


def _load_generated_python_dependencies() -> list[str]:
    project_data = _load_pyproject()
    dependencies = project_data.get("tool", {}).get("cvxgenrust", {}).get(
        "python-dependencies",
        ["cvxpy>=1.9", "numpy>=1.26"],
    )
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        raise TypeError("[tool.cvxgenrust].python-dependencies must be a list of strings")
    return dependencies


def _load_tool_config_string(name: str, default: str) -> str:
    project_data = _load_pyproject()
    return str(project_data.get("tool", {}).get("cvxgenrust", {}).get(name, default))


GENERATOR_VERSION = _load_generator_version()
CLARABEL_VERSION = _load_clarabel_version()
GENERATED_REQUIRES_PYTHON = _load_tool_config_string("generated-requires-python", ">=3.12")
MATURIN_VERSION = _load_tool_config_string("maturin-version", ">=1.7,<2")
PYO3_VERSION = _load_tool_config_string("pyo3-version", "0.25")
GENERATED_PYTHON_DEPENDENCIES = _load_generated_python_dependencies()


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
    offset: int


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


def _snake_case(name: str) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower() or "cgr_solver"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_solver"
    return candidate


def _python_package_name(name: str) -> str:
    candidate = _snake_case(name)
    if candidate[0].isdigit():
        candidate = f"cgr_{candidate}"
    return candidate


def _python_project_name(package_name: str) -> str:
    return package_name.replace("_", "-")


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_string_list(values: list[str], indent: str = "    ") -> str:
    return "\n".join(f"{indent}{_toml_string(value)}," for value in values)


def _python_install_command(target_dir: Path, project_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--target",
        str(target_dir),
        "--upgrade",
        str(project_dir),
    ]


def _ensure_pip_available() -> None:
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if pip_check.returncode == 0:
        return

    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "automatic wrapper compilation requires pip; install pip or run with wrapper=False"
        ) from error


def _compile_python_wrapper(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    python_source_dir = output_dir / "python"
    python_source_dir.mkdir(parents=True, exist_ok=True)
    _ensure_pip_available()
    subprocess.run(
        _python_install_command(python_source_dir, output_dir),
        cwd=output_dir,
        check=True,
    )


def _load_template(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


def _fill_template(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace(f"__{key}__", value)
    return template


def _comment_block(prefix: str, lines: list[str]) -> str:
    return "\n".join(f"{prefix} {line}" if line else prefix.rstrip() for line in lines)


def _generated_header_lines(artifact: str, generated_at: str, module_name: str | None = None) -> list[str]:
    lines = [
        f"Generated by {GENERATOR_DISPLAY_NAME} v{GENERATOR_VERSION}",
        f"Generated at: {generated_at}",
        f"Artifact: {artifact}",
    ]
    if module_name is not None:
        lines.insert(3, f"Problem module: {module_name}")
    return lines


def _generated_header(prefix: str, artifact: str, generated_at: str, module_name: str | None = None) -> str:
    lines = _generated_header_lines(artifact, generated_at, module_name=module_name)
    if not prefix:
        return "\n".join(lines)
    return _comment_block(prefix, lines)


def _rust_ident(name: str) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    if not candidate:
        candidate = "generated"
    if candidate[0].isdigit():
        candidate = f"p_{candidate}"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_value"
    return candidate


def _rust_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _rust_usize_slice(values: list[int] | tuple[int, ...]) -> str:
    return "&[" + ", ".join(f"{value}usize" for value in values) + "]"


def _rust_usize_vec(values: list[int]) -> str:
    return "vec![" + ", ".join(f"{value}usize" for value in values) + "]"


def _rust_f64(value: float) -> str:
    if math.isnan(value):
        raise ValueError("NaN coefficients are not supported")
    if math.isinf(value):
        return "f64::INFINITY" if value > 0 else "f64::NEG_INFINITY"
    if value == 0.0:
        return "0.0"
    return repr(float(value))


def _rust_f64_vec(values: list[float]) -> str:
    return "vec![" + ", ".join(_rust_f64(value) for value in values) + "]"


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
        module_name=_snake_case(module_name),
        parameter_vec_len=parameter_vec_len,
        cone_dims=dims,
        parameters=parameters,
        variables=variables,
        dual_variables=dual_variables,
        p_map=p_map,
        a_map=a_map,
        q_map=linear_obj_map,
    )


def _render_csr(name: str, spec: CsrMatrixSpec) -> str:
    return f"""let {name} = CsrMatrix {{
        rows: {spec.rows}usize,
        cols: {spec.cols}usize,
        indptr: {_rust_usize_vec(spec.indptr)},
        indices: {_rust_usize_vec(spec.indices)},
        data: {_rust_f64_vec(spec.data)},
    }};"""


def _render_pattern(name: str, spec: MatrixPatternSpec) -> str:
    return f"""let {name} = MatrixPattern {{
        rows: {spec.rows}usize,
        cols: {spec.cols}usize,
        indices: {_rust_usize_vec(spec.indices)},
        indptr: {_rust_usize_vec(spec.indptr)},
    }};"""


def _render_parameter_info(parameter: ParameterSpec) -> str:
    return f"""ParameterInfo {{
        name: {_rust_string(parameter.name)},
        shape: {_rust_usize_slice(parameter.shape)},
        size: {parameter.size}usize,
        offset: {parameter.offset}usize,
    }}"""


def _render_variable_info(variable: VariableSpec) -> str:
    return f"""VariableInfo {{
        name: {_rust_string(variable.name)},
        shape: {_rust_usize_slice(variable.shape)},
        size: {variable.size}usize,
        offset: {variable.offset}usize,
    }}"""


def _render_dual_variable_info(dual_variable: DualVariableSpec) -> str:
    return f"""DualVariableInfo {{
        name: {_rust_string(dual_variable.name)},
        shape: {_rust_usize_slice(dual_variable.shape)},
        size: {dual_variable.size}usize,
        offset: {dual_variable.offset}usize,
    }}"""


def _render_cone_dims(spec: ConeDimsSpec) -> str:
    return f"""ConeDims {{
            zero: {spec.zero}usize,
            nonneg: {spec.nonneg}usize,
            exp: {spec.exp}usize,
            soc: {_rust_usize_vec(spec.soc)},
            psd: {_rust_usize_vec(spec.psd)},
            p3d: {_rust_f64_vec(spec.p3d)},
        }}"""


def _render_generated_lib(spec: ProblemSpec, generated_at: str) -> str:
    parameters = ",\n    ".join(_render_parameter_info(parameter) for parameter in spec.parameters)
    variables = ",\n    ".join(_render_variable_info(variable) for variable in spec.variables)
    dual_variables = ",\n    ".join(
        _render_dual_variable_info(dual_variable)
        for dual_variable in spec.dual_variables
    )

    param_setters = []
    for parameter in spec.parameters:
        ident = _rust_ident(parameter.name)
        param_setters.append(
            f"""    pub fn set_{ident}(&mut self, value: &[f64]) -> Result<(), RuntimeError> {{
        self.set_parameter({_rust_string(parameter.name)}, value)
    }}

    pub fn update_{ident}(&mut self, index: usize, value: f64) -> Result<(), RuntimeError> {{
        self.update_parameter_entry({_rust_string(parameter.name)}, index, value)
    }}
"""
        )

    variable_getters = []
    for variable in spec.variables:
        ident = _rust_ident(variable.name)
        variable_getters.append(
            f"""    pub fn extract_{ident}(&self, solution: &[f64]) -> Result<Vec<f64>, RuntimeError> {{
        self.extract_variable({_rust_string(variable.name)}, solution)
    }}
"""
        )

    dual_variable_getters = []
    for dual_variable in spec.dual_variables:
        ident = _rust_ident(dual_variable.name)
        dual_variable_getters.append(
            f"""    pub fn extract_{ident}(&self, dual_solution: &[f64]) -> Result<Vec<f64>, RuntimeError> {{
        self.extract_dual_variable({_rust_string(dual_variable.name)}, dual_solution)
    }}
"""
        )

    cone_push_lines = []
    if spec.cone_dims.zero > 0:
        cone_push_lines.append(
            "        cones.push(SupportedConeT::<f64>::ZeroConeT(canonical.cones.zero));"
        )
    if spec.cone_dims.nonneg > 0:
        cone_push_lines.append(
            "        cones.push(SupportedConeT::<f64>::NonnegativeConeT(canonical.cones.nonneg));"
        )
    if spec.cone_dims.soc:
        cone_push_lines.append(
            "        for dim in &canonical.cones.soc {\n            cones.push(SupportedConeT::<f64>::SecondOrderConeT(*dim));\n        }"
        )
    if spec.cone_dims.psd:
        cone_push_lines.append(
            "        for dim in &canonical.cones.psd {\n            cones.push(SupportedConeT::<f64>::PSDTriangleConeT(*dim));\n        }"
        )
    if spec.cone_dims.exp > 0:
        cone_push_lines.append(
            "        for _ in 0..canonical.cones.exp {\n            cones.push(SupportedConeT::<f64>::ExponentialConeT());\n        }"
        )
    if spec.cone_dims.p3d:
        cone_push_lines.append(
            "        for alpha in &canonical.cones.p3d {\n            cones.push(SupportedConeT::<f64>::PowerConeT(*alpha));\n        }"
        )
    cone_push_code = "\n".join(cone_push_lines)
    canonical_method = _fill_template(
        _load_template("clarabel_canonical_method.rs.tmpl"),
        P_REDUCED=_render_csr("p_reduced", spec.p_map.reduced),
        P_PATTERN=_render_pattern("p_pattern", spec.p_map.pattern),
        A_REDUCED=_render_csr("a_reduced", spec.a_map.reduced),
        A_PATTERN=_render_pattern("a_pattern", spec.a_map.pattern),
        Q_REDUCED=_render_csr("c_reduced", spec.q_map.reduced),
        Q_OUTPUT_LEN=str(spec.q_map.output_len),
        CONE_DIMS=_render_cone_dims(spec.cone_dims),
    )
    solve_method = _fill_template(
        _load_template("clarabel_solve_method.rs.tmpl"),
        CONE_PUSH_CODE=cone_push_code,
    )
    python_methods = _fill_template(
        _load_template("pyo3_module.rs.tmpl"),
        LIB_NAME=spec.module_name.replace("-", "_"),
        PARAMETER_VEC_LEN=str(spec.parameter_vec_len),
        PARAMETER_VEC_LEN_MINUS_ONE=str(spec.parameter_vec_len - 1),
    )

    return _fill_template(
        _load_template("cgr_lib.rs.tmpl"),
        RUNTIME_RS=_fill_template(
            _load_template("runtime.rs.tmpl"),
            HEADER=_generated_header(
                "//",
                "Rust solver backend with canonicalization maps, Clarabel integration, and Python bindings",
                generated_at,
                module_name=spec.module_name,
            ),
        ),
        PARAMETER_COUNT=str(len(spec.parameters)),
        PARAMETERS=parameters,
        VARIABLE_COUNT=str(len(spec.variables)),
        VARIABLES=variables,
        DUAL_VARIABLE_COUNT=str(len(spec.dual_variables)),
        DUAL_VARIABLES=dual_variables,
        PARAMETER_VEC_LEN_MINUS_ONE=str(spec.parameter_vec_len - 1),
        CANONICAL_METHOD=canonical_method,
        SOLVE_METHOD=solve_method,
        PYTHON_METHODS=python_methods,
        PARAM_SETTERS="".join(param_setters),
        VARIABLE_GETTERS="".join(variable_getters),
        DUAL_VARIABLE_GETTERS="".join(dual_variable_getters),
    )


def _render_generated_pyproject(spec: ProblemSpec, package_name: str, generated_at: str) -> str:
    return _fill_template(
        _load_template("pyproject.toml.tmpl"),
        HEADER=_generated_header(
            "#",
            "Python build manifest for compiling the generated PyO3 extension with maturin",
            generated_at,
            module_name=spec.module_name,
        ),
        PROJECT_NAME=_python_project_name(package_name),
        PACKAGE_NAME=package_name,
        MODULE_NAME=spec.module_name,
        LIB_NAME=spec.module_name.replace("-", "_"),
        REQUIRES_PYTHON=GENERATED_REQUIRES_PYTHON,
        MATURIN_VERSION=MATURIN_VERSION,
        PYTHON_DEPENDENCIES=_toml_string_list(GENERATED_PYTHON_DEPENDENCIES),
    )


def _render_generated_init(spec: ProblemSpec, generated_at: str) -> str:
    return _generated_header(
        "#",
        "Generated Python solver module init file",
        generated_at,
        module_name=spec.module_name,
    ) + "\n"


def _render_generated_license(spec: ProblemSpec, generated_at: str) -> str:
    return _fill_template(
        _load_template("LICENSE.tmpl"),
        HEADER=_generated_header(
            "",
            "License for the generated solver support code",
            generated_at,
            module_name=spec.module_name,
        ),
    )


def _render_generated_cargo(spec: ProblemSpec, generated_at: str) -> str:
    if spec.cone_dims.psd:
        dependencies = """[target.'cfg(target_os = "macos")'.dependencies]
clarabel = { version = "__CLARABEL_VERSION__", features = ["sdp-accelerate"] }

[target.'cfg(not(target_os = "macos"))'.dependencies]
clarabel = { version = "__CLARABEL_VERSION__", features = ["sdp-openblas"] }"""
    else:
        dependencies = 'clarabel = "__CLARABEL_VERSION__"'
    return _fill_template(
        _load_template("Cargo.toml.tmpl"),
        HEADER=_generated_header(
            "#",
            "Cargo manifest for the generated solver crate",
            generated_at,
            module_name=spec.module_name,
        ),
        CRATE_NAME=spec.module_name,
        LIB_NAME=spec.module_name.replace("-", "_"),
        DEPENDENCIES=dependencies.replace("__CLARABEL_VERSION__", CLARABEL_VERSION),
        PYO3_VERSION=PYO3_VERSION,
    )


def _render_generated_readme(
    spec: ProblemSpec, package_name: str, output_dir_name: str, generated_at: str
) -> str:
    def code(value: str) -> str:
        return html.escape(value, quote=False)

    def attr(value: str) -> str:
        return html.escape(value, quote=True)

    def dimension(shape: tuple[int, ...], size: int) -> str:
        if not shape:
            return "scalar"
        shape_text = " by ".join(str(item) for item in shape)
        return f"{shape_text} ({size})" if size != math.prod(shape) else shape_text

    parameter_rows = "\n".join(
        "          <tr>"
        f"<td><code>{code(parameter.name)}</code></td>"
        f"<td>{code(dimension(parameter.shape, parameter.size))}</td>"
        f"<td>{parameter.size}</td>"
        f"<td>{parameter.offset}</td>"
        "</tr>"
        for parameter in spec.parameters
    ) or '          <tr><td colspan="4">No parameters</td></tr>'
    variable_rows = "\n".join(
        "          <tr>"
        f"<td><code>{code(variable.name)}</code></td>"
        f"<td>{code(dimension(variable.shape, variable.size))}</td>"
        f"<td>{variable.size}</td>"
        f"<td>{variable.offset}</td>"
        "</tr>"
        for variable in spec.variables
    ) or '          <tr><td colspan="4">No variables</td></tr>'
    dual_variable_rows = "\n".join(
        "          <tr>"
        f"<td><code>{code(dual_variable.name)}</code></td>"
        f"<td>{code(dimension(dual_variable.shape, dual_variable.size))}</td>"
        f"<td>{dual_variable.size}</td>"
        f"<td>{dual_variable.offset}</td>"
        "</tr>"
        for dual_variable in spec.dual_variables
    ) or '          <tr><td colspan="4">No canonical dual blocks</td></tr>'
    setter_lines = "\n".join(
        f"problem.set_{_rust_ident(parameter.name)}(&vec![0.0; {parameter.size}])?;"
        for parameter in spec.parameters
    )
    value_lines = ""
    if spec.variables:
        variable = spec.variables[0]
        ident = _rust_ident(variable.name)
        value_lines = (
            f"\nlet {ident} = problem.extract_{ident}(&solution.x)?;\n"
            f'println!("value = {{:?}}", {ident});'
        )
    rust_usage = _fill_template(
        """use __MODULE_NAME__::CGRProblem;

let mut problem = CGRProblem::new();
__SETTER_LINES__

let solution = problem.solve()?;
println!("status = {}", solution.status);
println!("objective = {}", solution.obj_val);__VALUE_LINES__
""",
        MODULE_NAME=spec.module_name,
        SETTER_LINES=setter_lines,
        VALUE_LINES=value_lines,
    ).strip()
    python_usage = _fill_template(
        """# Assume `problem` and the listed CVXPY parameters/variables are already defined.
import numpy as np
from __PACKAGE_NAME__.cgr_solver import cgr_solve

problem.register_solve("CGR", cgr_solve)
__PARAM_ASSIGNMENTS__

problem.solve(
    method="CGR",
    updated_params=[
        __UPDATED_PARAMS__
    ]
)
print(problem.status)
""",
        PACKAGE_NAME=package_name,
        PARAM_ASSIGNMENTS="\n".join(
            f"{parameter.name}.value = "
            + (
                "0.0"
                if parameter.size == 1
                else f"np.zeros({repr(tuple(parameter.shape)) if len(parameter.shape) != 1 else f'({parameter.shape[0]},)'})"
            )
            for parameter in spec.parameters
        ),
        UPDATED_PARAMS=",\n        ".join(repr(parameter.name) for parameter in spec.parameters),
    ).strip()
    generated_setters = "\n".join(
        f"    pub fn set_{_rust_ident(parameter.name)}(&mut self, value: &[f64]) -> Result<(), RuntimeError>;\n"
        f"    pub fn update_{_rust_ident(parameter.name)}(&mut self, index: usize, value: f64) -> Result<(), RuntimeError>;"
        for parameter in spec.parameters
    ) or "    // No generated parameter setters: this problem has no parameters."
    generated_extractors = "\n".join(
        f"    pub fn extract_{_rust_ident(variable.name)}(&self, solution: &[f64]) -> Result<Vec<f64>, RuntimeError>;"
        for variable in spec.variables
    ) or "    // No generated variable extractors."
    generated_dual_extractors = "\n".join(
        f"    pub fn extract_{_rust_ident(dual_variable.name)}(&self, dual_solution: &[f64]) -> Result<Vec<f64>, RuntimeError>;"
        for dual_variable in spec.dual_variables
    ) or "    // No generated dual block extractors."
    rust_api_snippet = _fill_template(
        """pub struct CGRProblem { ... }

impl CGRProblem {
    // Construct a problem object with zero-filled parameters and quiet Clarabel settings.
    pub fn new() -> Self;

    // Static metadata for generated parameter, primal variable, and canonical dual layouts.
    pub fn parameter_info(&self) -> &'static [ParameterInfo];
    pub fn variable_info(&self) -> &'static [VariableInfo];
    pub fn dual_variable_info(&self) -> &'static [DualVariableInfo];

    // Flattened parameter vector in CVXPY canonical order, including the trailing constant slot.
    pub fn parameter_vector(&self) -> Vec<f64>;

    // Clarabel settings accessors and common convenience setters.
    pub fn solver_settings(&self) -> &clarabel::solver::DefaultSettings<f64>;
    pub fn solver_settings_mut(&mut self) -> &mut clarabel::solver::DefaultSettings<f64>;
    pub fn set_solver_default_settings(&mut self);
    pub fn set_solver_verbose(&mut self, verbose: bool);
    pub fn set_solver_max_iter(&mut self, max_iter: u32);
    pub fn set_solver_time_limit(&mut self, time_limit: f64);
    pub fn set_solver_tol_gap_abs(&mut self, tol_gap_abs: f64);
    pub fn set_solver_tol_gap_rel(&mut self, tol_gap_rel: f64);
    pub fn set_solver_tol_feas(&mut self, tol_feas: f64);

    // Generic and generated parameter setters. Entry updates use zero-based flattened indices.
    pub fn set_parameter(&mut self, name: &str, value: &[f64]) -> Result<(), RuntimeError>;
    pub fn update_parameter_entry(&mut self, name: &str, index: usize, value: f64) -> Result<(), RuntimeError>;
__GENERATED_SETTERS__

    // Build the current canonical cone program data, or solve it with Clarabel.
    pub fn canonical_cone_prob(&self) -> Result<CanonicalConeQp, RuntimeError>;
    pub fn solve(&self) -> Result<SolveResult, RuntimeError>;

    // Extract primal variables from SolveResult.x.
    pub fn extract_variable(&self, name: &str, solution: &[f64]) -> Result<Vec<f64>, RuntimeError>;
__GENERATED_EXTRACTORS__

    // Extract canonical dual blocks from SolveResult.z. These are not always one-to-one
    // with original CVXPY constraints if canonicalization split a constraint.
    pub fn extract_dual_variable(&self, name: &str, dual_solution: &[f64]) -> Result<Vec<f64>, RuntimeError>;
__GENERATED_DUAL_EXTRACTORS__
}

pub struct SolveResult {
    pub x: Vec<f64>,      // primal solution vector
    pub z: Vec<f64>,      // canonical dual solution vector
    pub s: Vec<f64>,      // slack vector
    pub status: String,
    pub obj_val: f64,
    pub iterations: u32,
    pub r_prim: f64,
    pub r_dual: f64,
}""",
        GENERATED_SETTERS=generated_setters,
        GENERATED_EXTRACTORS=generated_extractors,
        GENERATED_DUAL_EXTRACTORS=generated_dual_extractors,
    )
    return _fill_template(
        _load_template("cgr_README.html.tmpl"),
        MODULE_NAME=attr(spec.module_name),
        MODULE_NAME_TEXT=code(spec.module_name),
        PACKAGE_NAME=attr(package_name),
        PACKAGE_NAME_TEXT=code(package_name),
        OUTPUT_DIR_NAME_TEXT=code(output_dir_name),
        GENERATED_AT=code(generated_at),
        GENERATOR_VERSION=code(GENERATOR_VERSION),
        PARAMETER_ROWS=parameter_rows,
        VARIABLE_ROWS=variable_rows,
        DUAL_VARIABLE_ROWS=dual_variable_rows,
        RUST_API_SNIPPET=code(rust_api_snippet),
        RUST_USAGE=code(rust_usage),
        PYTHON_USAGE=code(python_usage),
    )


def _render_generated_python_wrapper(spec: ProblemSpec, generated_at: str) -> str:
    parameter_entries = ",\n    ".join(
        repr(
            dict(
                name=parameter.name,
                shape=list(parameter.shape),
                size=parameter.size,
                offset=parameter.offset,
                pack=parameter.pack,
            )
        )
        for parameter in spec.parameters
    )
    return _fill_template(
        _load_template("cgr_solver.py.tmpl"),
        HEADER=_generated_header(
            "#",
            "CVXPY solve-method wrapper for the generated solver",
            generated_at,
            module_name=spec.module_name,
        ),
        PARAMETER_ENTRIES=parameter_entries,
        PARAMETER_VEC_LEN=str(spec.parameter_vec_len),
        LIB_NAME=spec.module_name.replace("-", "_"),
        SOLVER_METHOD_NAME=f"{spec.module_name}_cgr",
    )


def _render_generated_rust_example(spec: ProblemSpec, generated_at: str) -> str:
    param_setup = []
    for parameter in spec.parameters:
        ident = _rust_ident(parameter.name)
        param_setup.append(
            f"""    problem.set_{ident}(&vec![0.0; {parameter.size}])?;"""
        )

    variable_readback = ""
    if spec.variables:
        first_variable = spec.variables[0]
        first_ident = _rust_ident(first_variable.name)
        variable_readback = f"""
    let {first_ident} = problem.extract_{first_ident}(&solution.x)?;
    println!("{first_ident} = {{:?}}", {first_ident});"""

    solve_call = "problem.solve()?"

    return _fill_template(
        _load_template("cgr_solve.rs.tmpl"),
        HEADER=_generated_header(
            "//",
            "Rust example executable",
            generated_at,
            module_name=spec.module_name,
        ),
        MODULE_NAME=spec.module_name,
        PARAM_SETUP=chr(10).join(param_setup),
        SOLVE_CALL=solve_call,
        VARIABLE_READBACK=variable_readback,
    )


def generate_code(
    problem: cp.Problem,
    code_dir: str | Path,
    module_name: str,
    wrapper: bool = True,
    verbose: bool = True,
) -> GeneratedRustProject:
    output_dir = Path(code_dir)
    module = module_name
    package_name = _python_package_name(f"{module_name}_wrapper")

    def progress(message: str) -> None:
        if verbose:
            print(f"[CvxGenRust] {message}", file=sys.stderr)

    progress(f"Extracting problem data for module {module!r}")
    spec = extract_problem(problem, module_name=module)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    src_dir = output_dir / "src"
    package_dir = output_dir / "python" / package_name
    progress("Creating output directories")
    src_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)

    progress("Writing Rust crate and Python package files")
    (output_dir / "Cargo.toml").write_text(
        _render_generated_cargo(spec, generated_at), encoding="utf-8"
    )
    (output_dir / "pyproject.toml").write_text(
        _render_generated_pyproject(spec, package_name, generated_at), encoding="utf-8"
    )
    (output_dir / "LICENSE").write_text(
        _render_generated_license(spec, generated_at), encoding="utf-8"
    )
    (package_dir / "__init__.py").write_text(
        _render_generated_init(spec, generated_at), encoding="utf-8"
    )
    (src_dir / "lib.rs").write_text(_render_generated_lib(spec, generated_at), encoding="utf-8")
    (output_dir / "README.html").write_text(
        _render_generated_readme(spec, package_name, output_dir.name, generated_at), encoding="utf-8"
    )
    (package_dir / "cgr_solver.py").write_text(
        _render_generated_python_wrapper(spec, generated_at), encoding="utf-8"
    )
    examples_dir = output_dir / "examples"
    progress("Writing Rust example")
    examples_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "solve.rs").write_text(
        _render_generated_rust_example(spec, generated_at), encoding="utf-8"
    )

    if wrapper:
        progress("Compiling Python extension wrapper")
        _compile_python_wrapper(output_dir)

    progress(f"Done. Generated solver project at {output_dir}")
    return GeneratedRustProject(
        spec=spec,
        output_dir=output_dir,
    )
