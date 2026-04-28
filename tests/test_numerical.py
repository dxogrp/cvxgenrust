import numpy as np

from tests.support import GeneratedCodeTestCase


class NumericalTests(GeneratedCodeTestCase):
    def _solve_with_cvxpy_and_generated(self, fixture, module_name: str):
        cvxpy_value = fixture.problem.solve(solver="CLARABEL")
        cvxpy_solution = {
            name: np.array(variable.value, copy=True) for name, variable in fixture.variables.items()
        }

        tmpdir, method_name, _module = self._load_generated_module(fixture.problem, module_name)
        try:
            generated_value = fixture.problem.solve(method=method_name, warm_start=False)
        finally:
            self._clear_generated_module(tmpdir, method_name)
        return cvxpy_value, cvxpy_solution, generated_value

    def test_nonneg_ls_matches_cvxpy_solution(self):
        fixture = self._build_nonneg_ls_problem()
        cvxpy_value, cvxpy_solution, generated_value = self._solve_with_cvxpy_and_generated(
            fixture, "nonneg_ls"
        )

        self.assertAlmostEqual(float(generated_value), float(cvxpy_value), places=6)
        self.assertTrue(np.allclose(fixture.variables["x"].value, cvxpy_solution["x"], atol=1e-5))

    def test_box_qp_matches_cvxpy_solution(self):
        fixture = self._build_box_qp_problem()
        cvxpy_value, cvxpy_solution, generated_value = self._solve_with_cvxpy_and_generated(
            fixture, "box_qp"
        )

        self.assertAlmostEqual(float(generated_value), float(cvxpy_value), places=6)
        self.assertTrue(np.allclose(fixture.variables["x"].value, cvxpy_solution["x"], atol=1e-5))

    def test_socp_matches_cvxpy_solution(self):
        fixture = self._build_socp_problem()
        fixture.parameters["A"].value = np.array(
            [[1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.0, 0.1, 1.0], [1.0, 1.0, 1.0]]
        )
        fixture.parameters["b"].value = np.array([0.3, 0.2, 0.4, 0.1])
        fixture.parameters["rho"].value = 1.0
        cvxpy_value, cvxpy_solution, generated_value = self._solve_with_cvxpy_and_generated(
            fixture, "regularized_socp"
        )

        self.assertAlmostEqual(float(generated_value), float(cvxpy_value), places=6)
        self.assertTrue(np.allclose(fixture.variables["x"].value, cvxpy_solution["x"], atol=1e-5))
