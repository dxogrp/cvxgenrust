import subprocess
import tempfile

from cvxgenrust import cgr
from examples import actuator, mpc, network, sparse_nonneg_ls

from tests.support import GeneratedCodeTestCase


class FunctionalTests(GeneratedCodeTestCase):
    def test_sparse_nonneg_ls_matches_cvxpy(self):
        self._solve_with_generated_method(
            sparse_nonneg_ls.problem,
            "sparse_nonneg_ls",
            sparse_nonneg_ls.assign_example_data,
            sparse_nonneg_ls.x,
        )

    def test_actuator_matches_cvxpy(self):
        self._solve_with_generated_method(
            actuator.problem,
            "actuator",
            actuator.assign_example_data,
            actuator.u,
        )

    def test_mpc_matches_cvxpy(self):
        self._solve_with_generated_method(
            mpc.problem,
            "mpc",
            mpc.assign_example_data,
            mpc.U,
        )

    def test_network_matches_cvxpy(self):
        self._solve_with_generated_method(
            network.problem,
            "network",
            network.assign_example_data,
            network.f,
        )

    def test_generated_rust_example_runs(self):
        problem = self._build_problem()
        with tempfile.TemporaryDirectory() as tmpdir:
            cgr.generate_code(problem, code_dir=tmpdir, module_name="nonneg_ls")
            result = subprocess.run(
                ["cargo", "run", "--example", "solve"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("status = Solved", result.stdout)
