import contextlib
import io
import tempfile
import tomllib
from pathlib import Path

import cvxpy as cp
import pytest

from cvxgenrust import cgr
from cvxgenrust.config import (
    CLARABEL_VERSION,
    GENERATED_PYTHON_DEPENDENCIES,
    GENERATED_REQUIRES_PYTHON,
)
from cvxgenrust.extract import extract_problem
from cvxgenrust.generator import CodeGenerator
from cvxgenrust.names import (
    _python_distribution_name,
    _rust_ident,
    _rust_module_name,
    _snake_case,
    _wrapper_package_name,
)

from tests.support import GeneratedCodeTestCase


@pytest.mark.metadata
class MetadataTests(GeneratedCodeTestCase):
    def test_generated_name_normalization(self):
        self.assertEqual(_snake_case(""), "cgr_solver")
        self.assertEqual(_snake_case("class"), "class")
        self.assertEqual(_snake_case("Trace SDP"), "trace_sdp")
        self.assertEqual(_rust_module_name("123 solver"), "cgr_123_solver")
        self.assertEqual(_rust_module_name("crate"), "crate_solver")
        self.assertEqual(_rust_module_name("class"), "class_solver")
        self.assertEqual(_rust_ident("crate"), "crate_value")
        self.assertEqual(_rust_ident("class"), "class_value")
        self.assertEqual(_rust_ident("!!!"), "unnamed_value")
        self.assertEqual(_wrapper_package_name("Trace SDP"), "trace_sdp_wrapper")
        self.assertEqual(_wrapper_package_name("class"), "class_solver_wrapper")
        self.assertEqual(_python_distribution_name("trace_sdp_wrapper"), "trace-sdp-wrapper")

    def test_generate_code_uses_default_module_name(self):
        problem = self._build_nonneg_ls_problem().problem
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "custom_output"
            project = cgr.generate_code(problem, code_dir=output_dir, wrapper=False)

            self.assertEqual(project.spec.module_name, "cgr_module")
            self.assertEqual(project.output_dir, output_dir)
            self.assertIn(
                'module-name = "cgr_module_wrapper.cgr_module"',
                (output_dir / "pyproject.toml").read_text(encoding="utf-8"),
            )

    def test_generator_class_runs_generation_pipeline(self):
        problem = self._build_nonneg_ls_problem().problem
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "class_output"
            generator = CodeGenerator(module_name="class_solver", wrapper=False, verbose=False)

            project = generator.generate(problem, code_dir=output_dir)

            self.assertEqual(generator.package_name, "class_solver_wrapper")
            self.assertEqual(project.spec.module_name, "class_solver")
            self.assertEqual(project.output_dir, output_dir)
            self.assertTrue((output_dir / "src" / "lib.rs").exists())
            self.assertTrue(
                (output_dir / "python" / "class_solver_wrapper" / "cgr_solver.py").exists()
            )

    def test_generated_rust_method_name_clashes_raise(self):
        x = cp.Variable(name="variable")
        p1 = cp.Parameter(name="a-b")
        p2 = cp.Parameter(name="a b")
        p3 = cp.Parameter(name="parameter")
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p1 - p2 - p3)), [x >= 0])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "collision_output"
            with self.assertRaisesRegex(ValueError, "generated Rust API method name clash") as ctx:
                cgr.generate_code(
                    problem,
                    code_dir=output_dir,
                    module_name="collision",
                    wrapper=False,
                )

            message = str(ctx.exception)
            self.assertIn("'a-b' and 'a b'", message)
            self.assertIn("reserved Rust API name 'parameter'", message)
            self.assertIn("reserved Rust API name 'variable'", message)

    def test_extract_problem_metadata(self):
        spec = extract_problem(self._build_nonneg_ls_problem().problem, module_name="nonneg_ls")
        self.assertEqual(spec.module_name, "nonneg_ls")
        self.assertEqual(spec.parameter_vec_len, 10)
        self.assertEqual([parameter.name for parameter in spec.parameters], ["A", "b"])
        self.assertEqual(spec.parameters[0].size, 6)
        self.assertEqual([variable.name for variable in spec.variables], ["x"])
        self.assertEqual(
            [(dual.name, dual.size, dual.offset) for dual in spec.dual_variables],
            [("d0", 3, 0), ("d1", 2, 3)],
        )

    def test_generate_code_writes_crate(self):
        problem = self._build_nonneg_ls_problem().problem
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nonneg_ls_cgr"
            package_version = tomllib.loads(
                (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
            )["project"]["version"]
            progress = io.StringIO()
            with contextlib.redirect_stderr(progress):
                project = cgr.generate_code(problem, code_dir=output_dir, module_name="nonneg_ls", wrapper=False)
            progress_text = progress.getvalue()
            self.assertIn("[CvxGenRust] Extracting problem data", progress_text)
            self.assertIn("[CvxGenRust] Writing Rust crate and Python package files", progress_text)
            self.assertIn("[CvxGenRust] Done. Generated solver project", progress_text)
            cargo_toml = output_dir / "Cargo.toml"
            lib_rs = output_dir / "src" / "lib.rs"
            readme = output_dir / "README.html"
            license_file = output_dir / "LICENSE"
            package_dir = output_dir / "python" / "nonneg_ls_wrapper"
            wrapper = package_dir / "cgr_solver.py"
            pyproject = output_dir / "pyproject.toml"
            package_init = package_dir / "__init__.py"
            rust_example = output_dir / "examples" / "solve.rs"
            self.assertTrue(cargo_toml.exists())
            self.assertTrue(lib_rs.exists())
            self.assertTrue(readme.exists())
            self.assertTrue(license_file.exists())
            self.assertTrue(wrapper.exists())
            self.assertTrue(pyproject.exists())
            self.assertTrue(package_init.exists())
            self.assertTrue(rust_example.exists())
            self.assertEqual(project.spec.module_name, "nonneg_ls")
            cargo_text = cargo_toml.read_text(encoding="utf-8")
            lib_text = lib_rs.read_text(encoding="utf-8")
            self.assertIn("pub struct CGRProblem", lib_text)
            self.assertIn("def cgr_solve", wrapper.read_text(encoding="utf-8"))
            self.assertIn("pyo3", cargo_text)
            self.assertIn('version = "0.1.0"', cargo_text)
            self.assertIn('license = "Apache-2.0"', cargo_text)
            self.assertIn(f'clarabel = "{CLARABEL_VERSION}"', cargo_text)
            self.assertNotIn('features = ["sdp-accelerate"]', cargo_text)
            self.assertNotIn('features = ["sdp-openblas"]', cargo_text)
            self.assertIn('crate-type = ["rlib", "cdylib"]', cargo_text)
            pyproject_text = pyproject.read_text(encoding="utf-8")
            self.assertIn('name = "nonneg-ls-wrapper"', pyproject_text)
            self.assertIn('version = "0.1.0"', pyproject_text)
            self.assertIn('license = "Apache-2.0"', pyproject_text)
            self.assertIn(f'requires-python = "{GENERATED_REQUIRES_PYTHON}"', pyproject_text)
            self.assertIn('module-name = "nonneg_ls_wrapper.nonneg_ls"', pyproject_text)
            self.assertIn('python-source = "python"', pyproject_text)
            for dependency in GENERATED_PYTHON_DEPENDENCIES:
                self.assertIn(f'"{dependency}"', pyproject_text)
            self.assertNotIn("from .cgr_solver", package_init.read_text(encoding="utf-8"))
            self.assertIn("pub fn solve(&self)", lib_text)
            self.assertIn(f"Generated by CvxGenRust v{package_version}", lib_text)
            self.assertIn("Generated at:", lib_text)
            self.assertIn("wrap_pyfunction!(solve", lib_text)
            self.assertIn("pub fn set_a", lib_text)
            self.assertIn("pub fn update_a", lib_text)
            self.assertIn("pub fn extract_x", lib_text)
            self.assertIn("pub fn extract_d1", lib_text)
            self.assertIn("pub struct SolveResult", lib_text)
            self.assertIn("pub enum RuntimeError", lib_text)
            self.assertIn("pub fn solve_with_settings", lib_text)
            license_text = license_file.read_text(encoding="utf-8")
            self.assertIn("Apache License", license_text)
            self.assertIn("Generated CvxGenRust support code", license_text)
            self.assertIn("source optimization problem", license_text)
            self.assertIn("user-provided data", license_text)

    @pytest.mark.sdp
    def test_generate_code_enables_sdp_backend_for_psd_cones(self):
        problem = self._build_sdp_problem().problem
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "trace_sdp_cgr"
            cgr.generate_code(problem, code_dir=output_dir, module_name="trace_sdp", wrapper=False)
            cargo_text = (output_dir / "Cargo.toml").read_text(encoding="utf-8")

            self.assertIn('features = ["sdp-accelerate", "faer-sparse"]', cargo_text)
            self.assertIn('features = ["sdp-openblas", "faer-sparse"]', cargo_text)

    def test_extract_socp_problem_metadata(self):
        x = cp.Variable(3, name="x")
        A = cp.Parameter((4, 3), name="A")
        b = cp.Parameter(4, name="b")
        rho = cp.Parameter(nonneg=True, name="rho")
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.sum_squares(x)),
            [cp.norm(x, 2) <= rho, x >= 0, cp.sum(x) <= 1],
        )

        spec = extract_problem(problem, module_name="regularized_socp")
        self.assertEqual(spec.module_name, "regularized_socp")
        self.assertEqual(spec.cone_dims.soc, [4])

    def test_extract_parametric_quad_form_metadata(self):
        spec = extract_problem(
            self._build_parametric_quad_form_problem().problem,
            module_name="param_qp",
        )

        self.assertEqual(spec.module_name, "param_qp")
        self.assertEqual(
            [(parameter.name, parameter.size) for parameter in spec.parameters],
            [("P", 3), ("q", 2)],
        )
        self.assertEqual(spec.parameters[0].pack, "upper_tri")
        self.assertGreater(len(spec.p_map.reduced.data), 0)

    @pytest.mark.sdp
    def test_extract_sdp_problem_metadata(self):
        spec = extract_problem(self._build_sdp_problem().problem, module_name="trace_sdp")

        self.assertEqual(spec.module_name, "trace_sdp")
        self.assertEqual(spec.cone_dims.psd, [2])
        self.assertEqual(
            [
                (
                    variable.name,
                    variable.shape,
                    variable.size,
                    variable.canonical_size,
                    variable.unpack,
                )
                for variable in spec.variables
            ],
            [("X", (2, 2), 4, 3, "symmetric")],
        )
