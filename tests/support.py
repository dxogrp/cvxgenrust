import importlib.util
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import cvxpy as cp
import numpy as np

from cvxgenrust import cgr


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

    def _solve_with_generated_method(self, problem, module_name: str, assign_data, variable):
        assign_data()
        cvxpy_value = problem.solve(solver=cp.CLARABEL)
        cvxpy_x = np.array(variable.value, copy=True)

        tmpdir, method_name, _module = self._load_generated_module(problem, module_name)
        try:
            generated_value = problem.solve(
                method=method_name,
                warm_start=False,
            )
        finally:
            self._clear_generated_module(tmpdir, method_name)

        self.assertAlmostEqual(float(cvxpy_value), float(generated_value), places=5)
        self.assertTrue(np.allclose(cvxpy_x, variable.value, atol=1e-4))

    def _build_problem(self):
        m, n = 3, 2
        x = cp.Variable(n, name="x")
        A = cp.Parameter((m, n), name="A")
        b = cp.Parameter(m, name="b")
        problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])
        A.value = np.array([[1.0, 2.0], [0.0, 3.0], [0.0, 0.0]])
        b.value = np.array([1.0, 2.0, 3.0])
        return problem
