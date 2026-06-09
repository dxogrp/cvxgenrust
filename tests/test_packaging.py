import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pytest

from cvxgenrust.config import (
    CLARABEL_VERSION,
    GENERATED_PYTHON_DEPENDENCIES,
    GENERATED_REQUIRES_PYTHON,
    MATURIN_VERSION,
    PYO3_VERSION,
)
from tests.support import GeneratedCodeTestCase


@pytest.mark.metadata
class PackagingTests(GeneratedCodeTestCase):
    def _install_built_wheel(self, workspace: Path) -> tuple[Path, str]:
        project_root = Path(__file__).resolve().parent.parent
        package_version = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"]
        wheelhouse = workspace / "wheelhouse"
        site_dir = workspace / "site"
        wheelhouse.mkdir()

        subprocess.run(
            [
                "uv",
                "build",
                "--wheel",
                "--out-dir",
                str(wheelhouse),
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        wheels = sorted(wheelhouse.glob("cvxgenrust-*.whl"))
        self.assertEqual(len(wheels), 1)

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
                str(wheels[0]),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return site_dir, package_version

    def test_built_wheel_imports_without_source_pyproject(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            run_dir = workspace / "run"
            run_dir.mkdir()
            site_dir, package_version = self._install_built_wheel(workspace)

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from cvxgenrust.config import "
                        "CLARABEL_VERSION, GENERATED_REQUIRES_PYTHON, "
                        "MATURIN_VERSION, PYO3_VERSION, GENERATED_PYTHON_DEPENDENCIES, "
                        "GENERATOR_VERSION; "
                        "print(GENERATOR_VERSION, CLARABEL_VERSION, GENERATED_REQUIRES_PYTHON, "
                        "MATURIN_VERSION, PYO3_VERSION, ','.join(GENERATED_PYTHON_DEPENDENCIES))"
                    ),
                ],
                cwd=run_dir,
                check=True,
                capture_output=True,
                text=True,
                env={**self._cargo_env(), "PYTHONPATH": str(site_dir)},
            )

            self.assertEqual(
                result.stdout.strip(),
                " ".join(
                    [
                        package_version,
                        CLARABEL_VERSION,
                        GENERATED_REQUIRES_PYTHON,
                        MATURIN_VERSION,
                        PYO3_VERSION,
                        ",".join(GENERATED_PYTHON_DEPENDENCIES),
                    ]
                ),
            )

    def test_built_wheel_generate_code_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            run_dir = workspace / "run"
            run_dir.mkdir()
            site_dir, _ = self._install_built_wheel(workspace)

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "import cvxpy as cp; "
                        "import numpy as np; "
                        "import cvxgenrust as cgr; "
                        "A = cp.Parameter((2, 1), name='A'); "
                        "b = cp.Parameter(2, name='b'); "
                        "x = cp.Variable(1, name='x'); "
                        "A.value = np.array([[1.0], [2.0]]); "
                        "b.value = np.array([1.0, 2.0]); "
                        "problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0]); "
                        "project = cgr.generate_code("
                        "problem, code_dir='generated_solver', module_name='tiny_solver', "
                        "wrapper=False, verbose=False"
                        "); "
                        "root = Path(project.output_dir); "
                        "print(project.spec.module_name, "
                        "(root / 'Cargo.toml').exists(), "
                        "(root / 'pyproject.toml').exists(), "
                        "(root / 'README.html').exists())"
                    ),
                ],
                cwd=run_dir,
                check=True,
                capture_output=True,
                text=True,
                env={**self._cargo_env(), "PYTHONPATH": str(site_dir)},
            )

            self.assertEqual(result.stdout.strip(), "tiny_solver True True True")
