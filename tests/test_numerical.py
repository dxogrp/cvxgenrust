import numpy as np

from examples import actuator, regularized_socp

from tests.support import GeneratedCodeTestCase


class NumericalTests(GeneratedCodeTestCase):
    def test_nonneg_ls_known_solution(self):
        problem = self._build_problem()
        x = problem.variables()[0]
        tmpdir, method_name, _module = self._load_generated_module(problem, "nonneg_ls")
        try:
            value = problem.solve(method=method_name, warm_start=False)
        finally:
            self._clear_generated_module(tmpdir, method_name)

        expected_x = np.array([0.0, 8.0 / 13.0])
        expected_value = 118.0 / 13.0
        self.assertAlmostEqual(float(value), expected_value, places=6)
        self.assertTrue(np.allclose(x.value, expected_x, atol=1e-5))

    def test_actuator_known_solution(self):
        actuator.assign_example_data()
        tmpdir, method_name, _module = self._load_generated_module(actuator.problem, "actuator")
        try:
            value = actuator.problem.solve(method=method_name, warm_start=False)
        finally:
            self._clear_generated_module(tmpdir, method_name)

        expected_u = np.array(
            [
                0.3287581699,
                0.0934640523,
                0.3287581699,
                0.3287581699,
                0.0934640523,
                0.3287581699,
                0.0934640523,
                0.0934640523,
            ]
        )
        self.assertAlmostEqual(float(value), 0.4543790849673206, places=6)
        self.assertTrue(np.allclose(actuator.u.value, expected_u, atol=1e-5))

    def test_regularized_socp_known_solution(self):
        regularized_socp.A.value = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        )
        regularized_socp.b.value = np.array([0.3, 0.2, 0.4, 0.0])
        regularized_socp.rho.value = 1.0
        tmpdir, method_name, _module = self._load_generated_module(
            regularized_socp.problem, "regularized_socp"
        )
        try:
            value = regularized_socp.problem.solve(method=method_name, warm_start=False)
        finally:
            self._clear_generated_module(tmpdir, method_name)

        expected_x = np.array([6.7448662974e-02, 6.7321296282e-08, 1.5835774824e-01])
        self.assertAlmostEqual(float(value), 0.20642229086468347, places=6)
        self.assertTrue(np.allclose(regularized_socp.x.value, expected_x, atol=1e-5))

