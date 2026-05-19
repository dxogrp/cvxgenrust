import numpy as np
import pytest

from tests.support import GeneratedCodeTestCase


@pytest.mark.numerical
@pytest.mark.python_wrapper
class NumericalTests(GeneratedCodeTestCase):
    def _solve_with_cvxpy_and_generated(self, fixture, module_name: str):
        cvxpy_value = fixture.problem.solve(solver="CLARABEL")
        cvxpy_solution = {
            name: np.array(variable.value, copy=True) for name, variable in fixture.variables.items()
        }

        tmpdir, method_name, _module = self._load_generated_module(fixture.problem, module_name)
        try:
            generated_value = fixture.problem.solve(method=method_name)
            generated_solution = {
                name: np.array(variable.value, copy=True) for name, variable in fixture.variables.items()
            }
        finally:
            self._clear_generated_module(tmpdir, method_name)
        return cvxpy_value, cvxpy_solution, generated_value, generated_solution

    def test_nonneg_ls_matches_cvxpy_solution(self):
        fixture = self._build_nonneg_ls_problem()
        cvxpy_value, cvxpy_solution, generated_value, generated_solution = (
            self._solve_with_cvxpy_and_generated(
            fixture, "nonneg_ls"
            )
        )

        self.assertAlmostEqual(float(generated_value), float(cvxpy_value), places=6)
        self.assertTrue(np.allclose(generated_solution["x"], cvxpy_solution["x"], atol=1e-5))

    def test_box_qp_matches_cvxpy_solution(self):
        fixture = self._build_box_qp_problem()
        cvxpy_value, cvxpy_solution, generated_value, generated_solution = (
            self._solve_with_cvxpy_and_generated(
            fixture, "box_qp"
            )
        )

        self.assertAlmostEqual(float(generated_value), float(cvxpy_value), places=6)
        self.assertTrue(np.allclose(generated_solution["x"], cvxpy_solution["x"], atol=1e-5))

    def test_parametric_quad_form_matches_cvxpy_solution_after_update(self):
        fixture = self._build_parametric_quad_form_problem()
        tmpdir, method_name, _module = self._load_generated_module(fixture.problem, "param_qp")
        try:
            first_cvxpy_value = fixture.problem.solve(solver="CLARABEL")
            first_cvxpy_x = np.array(fixture.variables["x"].value, copy=True)
            first_cvxpy_duals = [
                np.array(constraint.dual_value, copy=True)
                for constraint in fixture.problem.constraints
            ]
            first_generated_value = fixture.problem.solve(method=method_name)
            first_generated_x = np.array(fixture.variables["x"].value, copy=True)
            first_generated_duals = [
                np.array(constraint.dual_value, copy=True)
                for constraint in fixture.problem.constraints
            ]

            fixture.parameters["P"].value = np.array([[0.75, 0.1], [0.1, 3.0]])
            second_cvxpy_value = fixture.problem.solve(solver="CLARABEL")
            second_cvxpy_x = np.array(fixture.variables["x"].value, copy=True)
            second_generated_value = fixture.problem.solve(
                method=method_name,
                updated_params=["P"],
            )
            second_generated_x = np.array(fixture.variables["x"].value, copy=True)
        finally:
            self._clear_generated_module(tmpdir, method_name)

        self.assertAlmostEqual(
            float(first_generated_value),
            float(first_cvxpy_value),
            places=6,
        )
        self.assertTrue(np.allclose(first_generated_x, first_cvxpy_x, atol=1e-5))
        for generated_dual, cvxpy_dual in zip(
            first_generated_duals,
            first_cvxpy_duals,
            strict=True,
        ):
            self.assertTrue(np.allclose(generated_dual, cvxpy_dual, atol=1e-5))
        self.assertAlmostEqual(
            float(second_generated_value),
            float(second_cvxpy_value),
            places=6,
        )
        self.assertTrue(np.allclose(second_generated_x, second_cvxpy_x, atol=1e-5))
        self.assertFalse(np.allclose(first_cvxpy_x, second_cvxpy_x, atol=1e-3))


    def test_socp_matches_cvxpy_solution(self):
        fixture = self._build_socp_problem()
        fixture.parameters["A"].value = np.array(
            [[1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.0, 0.1, 1.0], [1.0, 1.0, 1.0]]
        )
        fixture.parameters["b"].value = np.array([0.3, 0.2, 0.4, 0.1])
        fixture.parameters["rho"].value = 1.0
        cvxpy_value, cvxpy_solution, generated_value, generated_solution = (
            self._solve_with_cvxpy_and_generated(
            fixture, "regularized_socp"
            )
        )

        self.assertAlmostEqual(float(generated_value), float(cvxpy_value), places=6)
        self.assertTrue(np.allclose(generated_solution["x"], cvxpy_solution["x"], atol=1e-5))

    @pytest.mark.sdp
    def test_sdp_matches_cvxpy_solution(self):
        fixture = self._build_sdp_problem()
        cvxpy_value, cvxpy_solution, generated_value, generated_solution = (
            self._solve_with_cvxpy_and_generated(
            fixture, "trace_sdp"
            )
        )

        self.assertAlmostEqual(float(generated_value), float(cvxpy_value), places=6)
        self.assertTrue(np.allclose(generated_solution["X"], cvxpy_solution["X"], atol=1e-5))

    def test_flow_problem_matches_cvxpy_solution(self):
        fixture = self._build_flow_problem()
        cvxpy_value, cvxpy_solution, generated_value, generated_solution = (
            self._solve_with_cvxpy_and_generated(
            fixture, "network"
            )
        )

        self.assertAlmostEqual(float(generated_value), float(cvxpy_value), places=6)
        self.assertTrue(np.allclose(generated_solution["f"], cvxpy_solution["f"], atol=1e-5))
