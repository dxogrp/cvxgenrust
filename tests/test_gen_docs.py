import tempfile
from pathlib import Path

import pytest

from cvxgenrust import cgr

from tests.support import GeneratedCodeTestCase


@pytest.mark.metadata
class GeneratedDocsTests(GeneratedCodeTestCase):
    def _generate_nonneg_ls_project(self, tmpdir: str) -> Path:
        output_dir = Path(tmpdir) / "nonneg_ls_cgr"
        cgr.generate_code(
            self._build_nonneg_ls_problem().problem,
            code_dir=output_dir,
            module_name="nonneg_ls",
            wrapper=False,
        )
        return output_dir

    def test_generate_code_writes_rustdoc_for_solver_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = self._generate_nonneg_ls_project(tmpdir)

            lib_text = (output_dir / "src" / "lib.rs").read_text(encoding="utf-8")
            runtime_text = (output_dir / "src" / "runtime.rs").read_text(encoding="utf-8")
            readme_text = (output_dir / "README.html").read_text(encoding="utf-8")

            self.assertIn("/// Generated solver handle for this CVXPY problem.", lib_text)
            self.assertIn("/// Replaces parameter `A`.", lib_text)
            self.assertIn("/// Shape: 3 x 2. Flattened size: 6. Offset: 0.", lib_text)
            self.assertIn("pub fn set_a", lib_text)
            self.assertIn("/// Updates one scalar entry of parameter `A`.", lib_text)
            self.assertIn("pub fn update_a", lib_text)
            self.assertIn("/// Extracts primal variable `x` from `SolveResult.x`.", lib_text)
            self.assertIn("pub fn extract_x", lib_text)
            self.assertIn("/// Extracts canonical dual block `d1` from `SolveResult.z`.", lib_text)
            self.assertIn("pub fn extract_d1", lib_text)
            self.assertIn("/// Result returned by generated solve methods.", runtime_text)
            self.assertIn("pub struct SolveResult", runtime_text)
            self.assertIn("/// Errors returned by the generated Rust solver API.", runtime_text)
            self.assertIn("pub enum RuntimeError", runtime_text)
            self.assertIn("/// Solves the current problem with the provided Clarabel settings.", lib_text)
            self.assertIn("pub fn solve_with_settings", lib_text)
            self.assertIn("cargo doc --open", readme_text)

    def test_generate_code_writes_readme_documentation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = self._generate_nonneg_ls_project(tmpdir)

            readme_text = (output_dir / "README.html").read_text(encoding="utf-8")

            self.assertIn('problem.register_solve("CGR", cgr_solve)', readme_text)
            self.assertIn("from nonneg_ls_wrapper.cgr_solver import cgr_solve", readme_text)
            self.assertIn("generated project root", readme_text)
            self.assertIn("generated Python source directory", readme_text)
            self.assertIn("nonneg_ls_cgr/python", readme_text)
            self.assertIn("PYTHONPATH", readme_text)
            self.assertIn('method="CGR"', readme_text)
            self.assertIn("updated_params", readme_text)
            self.assertIn("<td>3 x 2</td>", readme_text)
            self.assertIn("cargo run --example solve", readme_text)
            self.assertIn("cargo doc --open", readme_text)
            self.assertIn("Cargo Dependency", readme_text)
            self.assertIn('nonneg_ls = { path = "/path/to/nonneg_ls_cgr" }', readme_text)
            self.assertIn("let solution = problem.solve()?;", readme_text)
            self.assertIn("Interface Reference", readme_text)
            self.assertIn("SolveResult", readme_text)
            self.assertIn("Entry updates use zero-based flattened indices", readme_text)
            self.assertIn("Canonical Dual Blocks", readme_text)
            self.assertIn("src/runtime.rs", readme_text)
            self.assertIn("src/data.rs", readme_text)
            self.assertIn("extract_d1", readme_text)
            self.assertIn("pub fn set_a", readme_text)
            self.assertIn("pub fn update_a", readme_text)
            self.assertIn("Solver Settings", readme_text)
            self.assertIn('<a href="#solver-settings">Solver Settings</a>', readme_text)
            self.assertIn("ClarabelSettings", readme_text)
            self.assertIn("use nonneg_ls::ClarabelSettings;", readme_text)
            self.assertIn("let solution = problem.solve_with_settings(settings)?;", readme_text)
            self.assertIn("max_iter=100", readme_text)
            self.assertIn("pub fn extract_x", readme_text)
