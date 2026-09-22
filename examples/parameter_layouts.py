import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dense and sparse parameter layouts

    This example solves the same least-squares problem through the generated
    Python wrapper and through direct Rust setters. It highlights the difference
    between logical CVXPY values and the packed slices accepted by Rust.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import shutil
    import sys
    import warnings

    import cvxpy as cp
    import cvxgenrust as cgr
    import marimo as mo
    import numpy as np
    from scipy import sparse

    return Path, cgr, cp, mo, np, shutil, sparse, sys, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the parameterized problem

    `M` and `b` are dense parameters. `S` has three entries that may change;
    every other entry is a structural zero. The target problem is

    \[
        \mathop{\mathrm{minimize}}_x \; \left\|(M + S)x - b\right\|_2^2.
    \]

    Python callers assign the logical values. Sparse coordinate/value pairs may
    be supplied in any order as long as they stay paired. CVXPY canonicalizes
    them into `S.sparse_idx` order, and the generated wrapper forwards the
    resulting `S.value_sparse.data`.
    """)
    return


@app.cell
def _(cp, np, sparse):
    x = cp.Variable(3, name="x")
    M = cp.Parameter((3, 3), name="M")

    sparse_rows = np.array([2, 0, 1])
    sparse_cols = np.array([1, 2, 0])
    S = cp.Parameter(
        (3, 3),
        sparsity=(sparse_rows, sparse_cols),
        name="S",
    )
    b = cp.Parameter(3, name="b")

    problem = cp.Problem(cp.Minimize(cp.sum_squares((M + S) @ x - b)))

    M.value = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.0, 1.5, 0.2],
            [0.3, 0.0, 1.25],
        ]
    )
    S.value_sparse = sparse.coo_array(
        (
            np.array([0.15, 0.4, -0.2]),
            (sparse_rows, sparse_cols),
        ),
        shape=S.shape,
    )
    b.value = np.array([0.95, 1.01, 0.475])
    expected_x = np.array([0.4, 0.7, 0.2])
    return expected_x, problem, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate and register the solver

    Code generation builds the Python extension and writes the Rust crate under
    `examples/generated/`. The companion Rust source `parameter_layouts.rs`
    located next to this notebook is copied into that crate so Cargo can run it
    as an example target.
    """)
    return


@app.cell
def _(Path, cgr, problem, shutil):
    example_dir = Path(__file__).resolve().parent
    project = cgr.generate_code(
        problem,
        code_dir=example_dir / "generated" / "parameter_layouts",
        module_name="parameter_layouts_cgr",
    )

    rust_example = project.output_dir / "examples" / "parameter_layouts.rs"
    shutil.copyfile(example_dir / "parameter_layouts.rs", rust_example)
    return project, rust_example


@app.cell
def _(problem, project, sys):
    _python_dir = str(project.output_dir / "python")
    if _python_dir not in sys.path:
        sys.path.insert(0, _python_dir)

    from parameter_layouts_cgr_wrapper.cgr_solver import cgr_solve

    problem.register_solve("CGR", cgr_solve)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solve through CVXPY and the generated wrapper
    """)
    return


@app.cell
def _(expected_x, np, problem, warnings, x):
    # CVXPY's reference solve reads the sparse expression through `.value`;
    # suppress only that known warning here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Reading from a sparse CVXPY expression via `\.value` is discouraged\..*",
            category=RuntimeWarning,
        )
        cvxpy_value = problem.solve(solver="CLARABEL")
    cvxpy_x = np.array(x.value, copy=True)

    generated_value = problem.solve(
        method="CGR",
        updated_params=["M", "S", "b"],
    )
    generated_x = np.array(x.value, copy=True)

    np.testing.assert_allclose(cvxpy_x, expected_x, atol=1e-5)
    np.testing.assert_allclose(generated_x, expected_x, atol=1e-5)
    np.testing.assert_allclose(generated_x, cvxpy_x, atol=1e-5)
    return cvxpy_value, cvxpy_x, generated_value, generated_x


@app.cell(hide_code=True)
def _(
    cvxpy_value,
    cvxpy_x,
    generated_value,
    generated_x,
    mo,
    np,
    project,
    rust_example,
):
    mo.md(f"""
    Both interfaces recover the expected solution.

    | solver | objective | x |
    | --- | ---: | --- |
    | CVXPY / Clarabel | `{cvxpy_value:.3g}` | `{np.array2string(cvxpy_x, precision=6)}` |
    | generated wrapper | `{generated_value:.3g}` | `{np.array2string(generated_x, precision=6)}` |

    Generated crate: `{project.output_dir}`

    Rust companion: `{rust_example}`

    The Rust companion supplies an already-packed column-major slice to `set_m`
    and the three canonical sparse values to `set_s`. From the repository root,
    run it with:

    ```bash
    cargo run --manifest-path examples/generated/parameter_layouts/Cargo.toml --example parameter_layouts
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
