import sys
import subprocess
import tempfile
from pathlib import Path

import pytest

from cvxgenrust import cgr

from tests.support import GeneratedCodeTestCase


class FunctionalTests(GeneratedCodeTestCase):
    def _write_nonneg_ls_problem_module(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "import cvxpy as cp",
                    "import numpy as np",
                    "",
                    "m, n = 3, 2",
                    'x = cp.Variable(n, name="x")',
                    'A = cp.Parameter((m, n), name="A")',
                    'b = cp.Parameter(m, name="b")',
                    "problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _write_rust_workflow_project(self, workspace: Path, generated_dir: Path) -> Path:
        project_dir = workspace / "rust_user_app"
        (project_dir / "src").mkdir(parents=True, exist_ok=True)
        (project_dir / "Cargo.toml").write_text(
            "\n".join(
                [
                    "[package]",
                    'name = "rust_user_app"',
                    'version = "0.1.0"',
                    'edition = "2024"',
                    "",
                    "[dependencies]",
                    f'nonneg_ls = {{ path = "{generated_dir.as_posix()}" }}',
                ]
            ),
            encoding="utf-8",
        )
        (project_dir / "src" / "main.rs").write_text(
            "\n".join(
                [
                    "use nonneg_ls::{CGRProblem, ClarabelSettings};",
                    "",
                    "fn main() -> Result<(), Box<dyn std::error::Error>> {",
                    "    let mut problem = CGRProblem::new();",
                    "    let mut settings = ClarabelSettings::<f64>::default();",
                    "    settings.max_iter = 100;",
                    "    settings.tol_feas = 1e-7;",
                    '    problem.set_a(&[1.0, 0.0, 0.0, 2.0, 3.0, 0.0])?;',
                    '    problem.set_b(&[1.0, 2.0, 3.0])?;',
                    "    problem.update_b(2, 3.0)?;",
                    "    let solution = problem.solve_with_settings(settings)?;",
                    '    let x = problem.extract_variable("x", &solution.x)?;',
                    '    let d1 = problem.extract_d1(&solution.z)?;',
                    '    println!("status = {}", solution.status);',
                    '    println!("objective = {}", solution.obj_val);',
                    '    println!("x = {:?}", x);',
                    '    println!("d1 = {:?}", d1);',
                    "    Ok(())",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        return project_dir

    def _write_structured_rust_project(self, workspace: Path, generated_dir: Path) -> Path:
        project_dir = workspace / "structured_rust_user_app"
        (project_dir / "src").mkdir(parents=True, exist_ok=True)
        (project_dir / "Cargo.toml").write_text(
            "\n".join(
                [
                    "[package]",
                    'name = "structured_rust_user_app"',
                    'version = "0.1.0"',
                    'edition = "2024"',
                    "",
                    "[dependencies]",
                    f'structured_parameters = {{ path = "{generated_dir.as_posix()}" }}',
                ]
            ),
            encoding="utf-8",
        )
        (project_dir / "src" / "main.rs").write_text(
            "\n".join(
                [
                    "use structured_parameters::{",
                    "    CGRProblem, ParameterLayout, SparseParameterPattern,",
                    "};",
                    "",
                    "fn assert_target(values: &[f64]) {",
                    "    let expected = [0.4, 0.7, 0.2];",
                    "    for (actual, expected) in values.iter().zip(expected) {",
                    '        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");',
                    "    }",
                    "}",
                    "",
                    "fn main() -> Result<(), Box<dyn std::error::Error>> {",
                    "    let mut problem = CGRProblem::new();",
                    "    let l = problem.parameter_info().iter().find(|p| p.name == \"L\").unwrap();",
                    "    assert_eq!(l.shape, &[3, 3]);",
                    "    assert_eq!(l.size, 6);",
                    "    assert!(matches!(",
                    "        &l.layout,",
                    "        ParameterLayout::Sparse(SparseParameterPattern::LowerTriangle)",
                    "    ));",
                    "    let s = problem.parameter_info().iter().find(|p| p.name == \"S\").unwrap();",
                    "    match &s.layout {",
                    "        ParameterLayout::Sparse(SparseParameterPattern::Explicit { flat_indices }) => {",
                    "            assert_eq!(*flat_indices, &[6, 1, 5]);",
                    "        }",
                    "        other => panic!(\"unexpected S layout: {other:?}\"),",
                    "    }",
                    "",
                    "    problem.set_l(&[2.0, 0.25, 1.5, -0.1, 0.3, 1.25])?;",
                    "    problem.set_s(&[0.4, -0.2, 0.15])?;",
                    "    problem.set_d(&[0.5, 0.75, 1.0])?;",
                    "    problem.set_b(&[1.08, 1.595, 0.725])?;",
                    "    let first = problem.solve()?;",
                    "    assert_target(&problem.extract_x(&first.x)?);",
                    "",
                    "    problem.set_l(&[1.8, 0.1, 1.4, 0.2, 0.25, 1.1])?;",
                    "    problem.set_s(&[0.2, -0.1, 0.05])?;",
                    "    problem.set_b(&[0.96, 1.505, 0.71])?;",
                    "    let second = problem.solve()?;",
                    "    assert_target(&problem.extract_x(&second.x)?);",
                    '    println!("structured parameter workflow passed");',
                    "    Ok(())",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        return project_dir

    @pytest.mark.python_wrapper
    def test_python_wrapper_user_workflow_runs(self):
        fixture = self._build_nonneg_ls_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            self._write_nonneg_ls_problem_module(workspace / "nonneg_ls.py")
            output_dir = workspace / "nonneg_ls_cgr"
            project = cgr.generate_code(fixture.problem, code_dir=output_dir, module_name="nonneg_ls")
            (workspace / "run_python_workflow.py").write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        "import sys",
                        "import numpy as np",
                        "",
                        "ROOT = Path(__file__).resolve().parent",
                        f"sys.path.insert(0, {str(project.output_dir / 'python')!r})",
                        "",
                        "from nonneg_ls_wrapper.cgr_solver import cgr_solve",
                        "from nonneg_ls import problem, A, b, x",
                        "",
                        'problem.register_solve("CGR", cgr_solve)',
                        "A.value = np.array([[1.0, 2.0], [0.0, 3.0], [0.0, 0.0]])",
                        "b.value = np.array([1.0, 2.0, 3.0])",
                        'value = problem.solve(method="CGR", updated_params=["A", "b"], max_iter=100, tol_feas=1e-7, verbose=False)',
                        'print("status =", problem.status)',
                        'print("value =", value)',
                        'print("x =", x.value)',
                    ]
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(workspace / "run_python_workflow.py")],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
                env=self._cargo_env(),
            )
            self.assertIn("status = optimal", result.stdout)
            self.assertIn("value =", result.stdout)
            self.assertIn("x =", result.stdout)

    @pytest.mark.python_wrapper
    def test_python_wrapper_rejects_invalid_and_pardiso_settings(self):
        fixture = self._build_nonneg_ls_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            self._write_nonneg_ls_problem_module(workspace / "nonneg_ls.py")
            output_dir = workspace / "nonneg_ls_cgr"
            project = cgr.generate_code(fixture.problem, code_dir=output_dir, module_name="nonneg_ls")
            (workspace / "run_python_errors.py").write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        "import sys",
                        "import numpy as np",
                        "",
                        "ROOT = Path(__file__).resolve().parent",
                        f"sys.path.insert(0, {str(project.output_dir / 'python')!r})",
                        "",
                        "from nonneg_ls_wrapper.cgr_solver import cgr_solve",
                        "from nonneg_ls import problem, A, b",
                        "",
                        'problem.register_solve("CGR", cgr_solve)',
                        "A.value = np.array([[1.0, 2.0], [0.0, 3.0], [0.0, 0.0]])",
                        "b.value = np.array([1.0, 2.0, 3.0])",
                        "for kwargs in ({'not_a_setting': 1}, {'pardiso_verbose': True}):",
                        "    try:",
                        "        problem.solve(method='CGR', updated_params=['A', 'b'], **kwargs)",
                        "    except (TypeError, RuntimeError) as error:",
                        "        print(type(error).__name__, error)",
                        "    else:",
                        "        raise AssertionError(f'expected TypeError for {kwargs}')",
                    ]
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(workspace / "run_python_errors.py")],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
                env=self._cargo_env(),
            )
            self.assertIn("unrecognized solver setting 'not_a_setting'", result.stdout)
            self.assertIn("unsupported Clarabel setting `pardiso_verbose`", result.stdout)


    @pytest.mark.python_wrapper
    def test_generated_python_package_installs(self):
        fixture = self._build_nonneg_ls_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_dir = workspace / "nonneg_ls_cgr"
            site_dir = workspace / "site"
            cgr.generate_code(fixture.problem, code_dir=output_dir, module_name="nonneg_ls", wrapper=False)
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    str(site_dir),
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=self._cargo_env(),
            )
            env = self._cargo_env()
            env["PYTHONPATH"] = str(site_dir)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from nonneg_ls_wrapper.cgr_solver import cgr_solve; from nonneg_ls_wrapper import nonneg_ls; print(cgr_solve.__name__, nonneg_ls.solve.__name__)",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn("cgr_solve solve", result.stdout)

    @pytest.mark.rust_smoke
    def test_rust_user_workflow_runs(self):
        fixture = self._build_nonneg_ls_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_dir = workspace / "nonneg_ls_cgr"
            cgr.generate_code(fixture.problem, code_dir=output_dir, module_name="nonneg_ls", wrapper=False)
            project_dir = self._write_rust_workflow_project(workspace, output_dir)
            result = subprocess.run(
                ["cargo", "run", "--manifest-path", str(project_dir / "Cargo.toml")],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
                env=self._cargo_env(),
            )
            self.assertIn("status = Solved", result.stdout)
            self.assertIn("objective =", result.stdout)
            self.assertIn("x =", result.stdout)

    @pytest.mark.rust_smoke
    def test_structured_parameter_rust_workflow_runs(self):
        fixture = self._build_structured_parameter_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_dir = workspace / "structured_parameters_cgr"
            cgr.generate_code(
                fixture.problem,
                code_dir=output_dir,
                module_name="structured_parameters",
                wrapper=False,
            )
            project_dir = self._write_structured_rust_project(workspace, output_dir)
            result = subprocess.run(
                ["cargo", "run", "--manifest-path", str(project_dir / "Cargo.toml")],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
                env=self._cargo_env(),
            )
            self.assertIn("structured parameter workflow passed", result.stdout)

    @pytest.mark.rust_smoke
    def test_generated_rust_example_runs(self):
        fixture = self._build_nonneg_ls_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_dir = workspace / "nonneg_ls_cgr"
            cgr.generate_code(fixture.problem, code_dir=output_dir, module_name="nonneg_ls", wrapper=False)
            result = subprocess.run(
                ["cargo", "run", "--example", "solve", "--manifest-path", str(output_dir / "Cargo.toml")],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
                env=self._cargo_env(),
            )
            self.assertIn("status = Solved", result.stdout)

    @pytest.mark.rust_smoke
    @pytest.mark.sdp
    def test_generated_rust_sdp_example_extracts_symmetric_variable(self):
        fixture = self._build_sdp_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_dir = workspace / "trace_sdp_cgr"
            cgr.generate_code(fixture.problem, code_dir=output_dir, module_name="trace_sdp", wrapper=False)
            result = subprocess.run(
                ["cargo", "run", "--example", "solve", "--manifest-path", str(output_dir / "Cargo.toml")],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
                env=self._cargo_env(),
            )
            self.assertIn("status = Solved", result.stdout)
            self.assertIn("x = [", result.stdout.lower())
