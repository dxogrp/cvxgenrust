import sys
import subprocess
import tempfile
from pathlib import Path

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
                    f'nonneg_ls = {{ path = "{generated_dir}" }}',
                ]
            ),
            encoding="utf-8",
        )
        (project_dir / "src" / "main.rs").write_text(
            "\n".join(
                [
                    "use nonneg_ls::CGRProblem;",
                    "",
                    "fn main() -> Result<(), Box<dyn std::error::Error>> {",
                    "    let mut problem = CGRProblem::new();",
                    '    problem.set_a(&[1.0, 0.0, 0.0, 2.0, 3.0, 0.0])?;',
                    '    problem.set_b(&[1.0, 2.0, 3.0])?;',
                    "    let solution = problem.solve()?;",
                    '    let x = problem.extract_variable("x", &solution.x)?;',
                    '    println!("status = {}", solution.status);',
                    '    println!("objective = {}", solution.obj_val);',
                    '    println!("x = {:?}", x);',
                    "    Ok(())",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        return project_dir

    def test_python_wrapper_user_workflow_runs(self):
        fixture = self._build_nonneg_ls_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            self._write_nonneg_ls_problem_module(workspace / "nonneg_ls.py")
            output_dir = workspace / "nonneg_ls_cgr"
            cgr.generate_code(fixture.problem, code_dir=output_dir, module_name="nonneg_ls")
            (workspace / "run_python_workflow.py").write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        "import sys",
                        "import numpy as np",
                        "",
                        "ROOT = Path(__file__).resolve().parent",
                        'sys.path.insert(0, str(ROOT / "nonneg_ls_cgr"))',
                        "",
                        "from cgr_solver import cgr_solve",
                        "from nonneg_ls import problem, A, b, x",
                        "",
                        'problem.register_solve("CGR", cgr_solve)',
                        "A.value = np.array([[1.0, 2.0], [0.0, 3.0], [0.0, 0.0]])",
                        "b.value = np.array([1.0, 2.0, 3.0])",
                        'value = problem.solve(method="CGR", updated_params=["A", "b"])',
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
            )
            self.assertIn("status = optimal", result.stdout)
            self.assertIn("value =", result.stdout)
            self.assertIn("x =", result.stdout)

    def test_rust_user_workflow_runs(self):
        fixture = self._build_nonneg_ls_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_dir = workspace / "nonneg_ls_cgr"
            cgr.generate_code(fixture.problem, code_dir=output_dir, module_name="nonneg_ls")
            project_dir = self._write_rust_workflow_project(workspace, output_dir)
            result = subprocess.run(
                ["cargo", "run", "--manifest-path", str(project_dir / "Cargo.toml")],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("status = Solved", result.stdout)
            self.assertIn("objective =", result.stdout)
            self.assertIn("x =", result.stdout)

    def test_generated_rust_example_runs(self):
        fixture = self._build_nonneg_ls_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_dir = workspace / "nonneg_ls_cgr"
            cgr.generate_code(fixture.problem, code_dir=output_dir, module_name="nonneg_ls")
            result = subprocess.run(
                ["cargo", "run", "--example", "solve", "--manifest-path", str(output_dir / "Cargo.toml")],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("status = Solved", result.stdout)
