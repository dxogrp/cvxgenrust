import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Nearest Correlation Matrix
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

    A common cleanup step for estimated covariance or correlation data is to project a
    noisy symmetric matrix onto the set of valid correlation matrices. The repaired
    matrix must be positive semidefinite with ones on the diagonal:
    \[
        \begin{array}{ll}
            \mbox{minimize} & \|X - S\|_F^2\\
            \mbox{subject to} & X \succeq 0\\
                              & \operatorname{diag}(X) = 1,
        \end{array}
    \]
    where \(S\) is the current noisy estimate and \(X\) is the repaired matrix.
    """)
    return


@app.cell
def _(cp, np):
    n = 20
    X = cp.Variable((n, n), symmetric=True, name="X")
    S = cp.Parameter((n, n), symmetric=True, name="S")
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(X - S)),
        [X >> 0, cp.diag(X) == 1],
    )

    rng = np.random.default_rng(7)
    factors = rng.normal(size=(n, 2))
    covariance = factors @ factors.T + 0.2 * np.eye(n)
    scale = np.sqrt(np.diag(covariance))
    base_corr = covariance / np.outer(scale, scale)

    perturbation = rng.normal(size=(n, n))
    perturbation = 0.5 * (perturbation + perturbation.T)
    np.fill_diagonal(perturbation, 0.0)
    S.value = base_corr
    return S, X, base_corr, n, perturbation, problem


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
        code_dir="generated/nearest_corr_mat",
        module_name="nearest_cov",
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
    sys.path.append("generated/nearest_corr_mat/python")
    from nearest_cov_wrapper.cgr_solver import cgr_solve

    problem.register_solve("CGR", cgr_solve)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example runs
    """)
    return


@app.cell
def _(S, X, base_corr, np, perturbation, problem, time):
    rows = []
    last_input = None
    last_repaired = None
    for noise in [0.0, 0.2, 0.4, 0.6]:
        noisy = base_corr + noise * perturbation
        noisy = 0.5 * (noisy + noisy.T)
        np.fill_diagonal(noisy, 1.0)
        S.value = noisy

        cvxpy_start = time.perf_counter()
        cvxpy_value = problem.solve(solver="CLARABEL")
        cvxpy_time = time.perf_counter() - cvxpy_start
        cvxpy_x = np.array(X.value, copy=True)

        generated_start = time.perf_counter()
        generated_value = problem.solve(method="CGR", updated_params=["S"])
        generated_time = time.perf_counter() - generated_start
        generated_x = np.array(X.value, copy=True)

        last_input = noisy
        last_repaired = generated_x
        rows.append(
            {
                "noise": noise,
                "input_min_eig": np.linalg.eigvalsh(noisy).min(),
                "repaired_min_eig": np.linalg.eigvalsh(generated_x).min(),
                "diag_error": np.linalg.norm(np.diag(generated_x) - 1.0, ord=np.inf),
                "cvxpy_value": cvxpy_value,
                "generated_value": generated_value,
                "cvxpy_time_ms": 1000.0 * cvxpy_time,
                "generated_time_ms": 1000.0 * generated_time,
                "objective_diff": abs(generated_value - cvxpy_value),
                "solution_diff": np.linalg.norm(generated_x - cvxpy_x),
            }
        )
    return last_input, last_repaired, rows


@app.cell
def _(last_input, last_repaired, mo, n, np, project, rows):
    table_rows = "\n".join(
        "| {noise:.2f} | {input_min_eig:.3g} | {repaired_min_eig:.3g} | {diag_error:.3g} | {cvxpy_value:.6g} | {generated_value:.6g} | {cvxpy_time_ms:.3f} | {generated_time_ms:.3f} | {objective_diff:.3g} | {solution_diff:.3g} |".format(
            **row
        )
        for row in rows
    )
    input_preview = np.array2string(last_input, precision=3, suppress_small=True)
    repaired_preview = np.array2string(last_repaired, precision=3, suppress_small=True)
    mo.md(f"""
    ## Parameter update sweep

    Generated crate: `{project.output_dir}`

    | noise scale | input min eig | repaired min eig | max diag error | CVXPY objective | generated objective | CVXPY time (ms) | generated time (ms) | objective diff | solution diff |
    | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    {table_rows}

    Fixed matrix size: `{n} x {n}`. Each row updates `S`, then reuses the same
    generated SDP solver. The final input and repaired matrices are:

    ```text
    input S =
    {input_preview}

    repaired X =
    {repaired_preview}
    ```
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
