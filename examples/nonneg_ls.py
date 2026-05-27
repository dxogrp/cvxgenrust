import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Nonnegative Least Squares
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import cvxgenrust as cgr

    import sys
    import time

    return cgr, cp, mo, np, sys, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    We consider a simple nonnegative least squares problem
    \[
        \begin{array}{ll}
            \mbox{minimize} & {\|Ax - b\|}_2^2\\
            \mbox{subject to} & x \succeq 0,
        \end{array}
    \]
    where $x \in \mathbf{R}^n$ is the variable, $A \in \mathbf{R}^{m \times n}$ and $b \in \mathbf{R}^m$ are (parameterized) problem data.
    """)
    return


@app.cell
def _(cp, np):
    m, n = 120, 20
    x = cp.Variable(n, name="x")
    A = cp.Parameter((m, n), name="A")
    b = cp.Parameter(m, name="b")
    problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])

    rng = np.random.default_rng(2)
    A_base = rng.normal(size=(m, n))
    x_true = rng.uniform(0.0, 1.0, size=n)
    noise_direction = rng.normal(size=m)
    A_direction = rng.normal(size=(m, n))
    A.value = A_base
    b.value = A_base @ x_true
    return A, A_base, A_direction, b, n, noise_direction, problem, x, x_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate solver code
    """)
    return


@app.cell
def _(cgr, problem):
    project = cgr.generate_code(
        problem,
        code_dir="generated/nonneg_ls",
        module_name="nonneg_ls_cgr"
    )
    return (project,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Register CVXPY solver
    """)
    return


@app.cell
def _(problem, sys):
    sys.path.append('generated/nonneg_ls/python')
    from nonneg_ls_cgr_wrapper.cgr_solver import cgr_solve
    problem.register_solve("CGR", cgr_solve)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example runs
    """)
    return


@app.cell
def _(
    A,
    A_base,
    A_direction,
    b,
    noise_direction,
    np,
    problem,
    time,
    x,
    x_true,
):
    rows = []
    last_residual = None
    last_x = None
    for noise in [0.0, 0.05, 0.1, 0.2]:
        A.value = A_base + noise * A_direction
        b.value = A.value @ x_true + noise * noise_direction

        cvxpy_start = time.perf_counter()
        cvxpy_value = problem.solve(solver="CLARABEL")
        cvxpy_time = time.perf_counter() - cvxpy_start
        cvxpy_x = np.array(x.value, copy=True)

        generated_start = time.perf_counter()
        generated_value = problem.solve(method="CGR", updated_params=["A", "b"])
        generated_time = time.perf_counter() - generated_start
        generated_x = np.array(x.value, copy=True)
        last_x = generated_x
        last_residual = A.value @ generated_x - b.value
        rows.append(
            {
                "noise": noise,
                "cvxpy_value": cvxpy_value,
                "generated_value": generated_value,
                "cvxpy_time_ms": 1000.0 * cvxpy_time,
                "generated_time_ms": 1000.0 * generated_time,
                "objective_diff": abs(generated_value - cvxpy_value),
                "solution_diff": np.linalg.norm(generated_x - cvxpy_x),
            }
        )
    return (rows,)


@app.cell
def _(mo, n, project, rows):
    table_rows = "\n".join(
        "| {noise:.2f} | {cvxpy_value:.6g} | {generated_value:.6g} | {cvxpy_time_ms:.3f} | {generated_time_ms:.3f} | {objective_diff:.3g} | {solution_diff:.3g} |".format(
            **row
        )
        for row in rows
    )
    mo.md(f"""
    ## Parameter update sweep

    Generated crate: `{project.output_dir}`

    | noise scale | CVXPY objective | generated objective | CVXPY time (ms) | generated time (ms) | objective diff | solution diff |
    | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    {table_rows}

    Fixed problem size: `{n}` variables. Each row updates `A` and `b`, then reuses the
    same generated solver.
    """)
    return


if __name__ == "__main__":
    app.run()
