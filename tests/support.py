import importlib
import os
import sys
import tempfile
import unittest
import uuid
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np

from cvxgenrust import cgr
from cvxgenrust.generator import _python_package_name


@dataclass
class ProblemFixture:
    problem: cp.Problem
    variables: dict[str, cp.Variable]
    parameters: dict[str, cp.Parameter]


class GeneratedCodeTestCase(unittest.TestCase):
    @staticmethod
    def _cargo_env():
        env = os.environ.copy()
        env.setdefault(
            "CARGO_TARGET_DIR",
            str(Path(tempfile.gettempdir()) / "cvxgenrust-cargo-target"),
        )
        return env

    def _load_generated_module(self, problem, module_name: str):
        unique_name = f"{module_name}_{uuid.uuid4().hex[:8]}"
        tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=(sys.platform == "win32"))
        os.environ.setdefault(
            "CARGO_TARGET_DIR",
            str(Path(tempfile.gettempdir()) / "cvxgenrust-cargo-target"),
        )
        project = cgr.generate_code(
            problem,
            code_dir=Path(tmpdir.name) / unique_name,
            module_name=unique_name,
        )
        python_source_dir = str(project.output_dir / "python")
        sys.path.insert(0, python_source_dir)
        package_name = _python_package_name(f"{project.spec.module_name}_wrapper")
        method_name = f"{unique_name}_cgr"
        module = importlib.import_module(f"{package_name}.cgr_solver")
        setattr(tmpdir, "cgr_python_source_dir", python_source_dir)
        setattr(tmpdir, "cgr_package_name", package_name)
        problem.register_solve(method_name, module.cgr_solve)
        return tmpdir, method_name, module

    def _clear_generated_module(self, tmpdir, method_name: str):
        sys.modules.pop(method_name, None)
        package_name = getattr(tmpdir, "cgr_package_name", None)
        if package_name is not None:
            for name in list(sys.modules):
                if name == package_name or name.startswith(f"{package_name}."):
                    sys.modules.pop(name, None)
        python_source_dir = getattr(tmpdir, "cgr_python_source_dir", None)
        if python_source_dir in sys.path:
            sys.path.remove(python_source_dir)
        tmpdir.cleanup()

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

    def _build_parametric_quad_form_problem(self):
        x = cp.Variable(2, name="x")
        P = cp.Parameter((2, 2), PSD=True, name="P")
        q = cp.Parameter(2, name="q")
        objective = cp.quad_form(x, P) + q @ x
        constraints = [cp.sum(x) == 1, x >= 0]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        P.value = np.array([[2.0, 0.25], [0.25, 1.0]])
        q.value = np.array([-1.0, -0.25])
        return ProblemFixture(problem=problem, variables={"x": x}, parameters={"P": P, "q": q})

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

    def _build_sdp_problem(self):
        X = cp.Variable((2, 2), symmetric=True, name="X")
        C = cp.Parameter((2, 2), symmetric=True, name="C")
        problem = cp.Problem(
            cp.Minimize(cp.sum(cp.multiply(C, X))),
            [X >> 0, cp.trace(X) == 1],
        )
        C.value = np.array([[1.0, 0.25], [0.25, 3.0]])
        return ProblemFixture(problem=problem, variables={"X": X}, parameters={"C": C})

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
