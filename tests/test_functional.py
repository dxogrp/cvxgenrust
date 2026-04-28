import subprocess
import tempfile

from cvxgenrust import cgr

from tests.support import GeneratedCodeTestCase


class FunctionalTests(GeneratedCodeTestCase):
    def test_sparse_nonneg_ls_matches_cvxpy(self):
        fixture = self._build_nonneg_ls_problem()
        self._solve_with_generated_method(
            fixture.problem,
            "sparse_nonneg_ls",
            fixture.variables,
        )

    def test_box_qp_matches_cvxpy(self):
        fixture = self._build_box_qp_problem()
        self._solve_with_generated_method(
            fixture.problem,
            "box_qp",
            fixture.variables,
        )

    def test_socp_matches_cvxpy(self):
        fixture = self._build_socp_problem()
        self._solve_with_generated_method(
            fixture.problem,
            "regularized_socp",
            fixture.variables,
        )

    def test_flow_problem_matches_cvxpy(self):
        fixture = self._build_flow_problem()
        self._solve_with_generated_method(
            fixture.problem,
            "network",
            fixture.variables,
        )

    def test_generated_rust_example_runs(self):
        fixture = self._build_nonneg_ls_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            cgr.generate_code(fixture.problem, code_dir=tmpdir, module_name="nonneg_ls")
            result = subprocess.run(
                ["cargo", "run", "--example", "solve"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("status = Solved", result.stdout)
