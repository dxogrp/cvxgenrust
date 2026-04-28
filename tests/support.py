import importlib.util
import sys
import tempfile
import unittest
import uuid
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np

from cvxgenrust import cgr


@dataclass
class ProblemFixture:
    problem: cp.Problem
    variables: dict[str, cp.Variable]
    parameters: dict[str, cp.Parameter]


class GeneratedCodeTestCase(unittest.TestCase):
    def _load_generated_module(self, problem, module_name: str):
        unique_name = f"{module_name}_{uuid.uuid4().hex[:8]}"
        tmpdir = tempfile.TemporaryDirectory()
        cgr.generate_code(problem, code_dir=tmpdir.name, module_name=unique_name)
        init_path = Path(tmpdir.name) / "__init__.py"
        method_name = f"{unique_name}_cgr"
        spec = importlib.util.spec_from_file_location(method_name, init_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[method_name] = module
        spec.loader.exec_module(module)
        module.register_solve_method(method_name)
        return tmpdir, method_name, module

    def _clear_generated_module(self, tmpdir, method_name: str):
        sys.modules.pop(method_name, None)
        tmpdir.cleanup()

    def _solve_with_generated_method(self, problem, module_name: str, variables):
        cvxpy_value = problem.solve(solver=cp.CLARABEL)
        cvxpy_solution = {
            name: np.array(variable.value, copy=True) for name, variable in variables.items()
        }

        tmpdir, method_name, _module = self._load_generated_module(problem, module_name)
        try:
            generated_value = problem.solve(
                method=method_name,
                warm_start=False,
            )
        finally:
            self._clear_generated_module(tmpdir, method_name)

        self.assertAlmostEqual(float(cvxpy_value), float(generated_value), places=5)
        for name, variable in variables.items():
            self.assertTrue(np.allclose(cvxpy_solution[name], variable.value, atol=1e-4), msg=name)

    def _build_nonneg_ls_problem(self):
        m, n = 3, 2
        x = cp.Variable(n, name="x")
        A = cp.Parameter((m, n), name="A")
        b = cp.Parameter(m, name="b")
        problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])
        A.value = np.array([[1.0, 2.0], [0.0, 3.0], [0.0, 0.0]])
        b.value = np.array([1.0, 2.0, 3.0])
        return ProblemFixture(problem=problem, variables={"x": x}, parameters={"A": A, "b": b})

    def _build_box_qp_problem(self):
        x = cp.Variable(3, name="x")
        x_ref = cp.Parameter(3, name="x_ref")
        q = cp.Parameter(3, name="q")
        A = cp.Parameter((1, 3), name="A")
        b = cp.Parameter(1, name="b")
        objective = 0.5 * cp.sum_squares(x - x_ref) + q @ x
        constraints = [A @ x == b, x >= 0, x <= 1]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        x_ref.value = np.array([0.8, 0.1, 0.6])
        q.value = np.array([-8.0, -3.0, -3.0])
        A.value = np.array([[1.0, 1.0, 1.0]])
        b.value = np.array([1.0])
        return ProblemFixture(
            problem=problem,
            variables={"x": x},
            parameters={"x_ref": x_ref, "q": q, "A": A, "b": b},
        )

    def _build_socp_problem(self):
        x = cp.Variable(3, name="x")
        A = cp.Parameter((4, 3), name="A")
        b = cp.Parameter(4, name="b")
        rho = cp.Parameter(nonneg=True, name="rho")
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.sum_squares(x)),
            [cp.norm(x, 2) <= rho, x >= 0, cp.sum(x) <= 1],
        )
        A.value = np.array(
            [[1.0, 0.2, 0.0], [0.0, 1.0, 0.1], [0.0, 0.3, 1.0], [1.0, 1.0, 1.0]]
        )
        b.value = np.array([0.4, 0.3, 0.5, 0.2])
        rho.value = 0.8
        return ProblemFixture(
            problem=problem,
            variables={"x": x},
            parameters={"A": A, "b": b, "rho": rho},
        )

    def _build_flow_problem(self):
        f = cp.Variable(3, name="f")
        c = cp.Parameter(3, name="c")
        d = cp.Parameter(2, name="d")
        problem = cp.Problem(
            cp.Minimize(c @ f),
            [
                f >= 0,
                f <= 3,
                f[0] + f[2] == d[0],
                f[0] + f[1] == d[1],
            ],
        )
        c.value = np.array([1.0, 2.0, 0.5])
        d.value = np.array([0.75, 1.25])
        return ProblemFixture(problem=problem, variables={"f": f}, parameters={"c": c, "d": d})
