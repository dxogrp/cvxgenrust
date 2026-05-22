import subprocess
import tempfile
from pathlib import Path

import pytest

from cvxgenrust import cgr

from tests.support import GeneratedCodeTestCase


@pytest.mark.rust_smoke
class GeneratedRustTests(GeneratedCodeTestCase):
    def test_generated_rust_crate_formats_and_checks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nonneg_ls_cgr"
            cgr.generate_code(
                self._build_nonneg_ls_problem().problem,
                code_dir=output_dir,
                module_name="nonneg_ls",
                wrapper=False,
            )

            manifest_path = output_dir / "Cargo.toml"
            subprocess.run(
                ["cargo", "fmt", "--check", "--manifest-path", str(manifest_path)],
                cwd=output_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["cargo", "check", "--manifest-path", str(manifest_path)],
                cwd=output_dir,
                check=True,
                capture_output=True,
                text=True,
                env=self._cargo_env(),
            )
