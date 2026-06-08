# cvxgenrust

`cvxgenrust` takes a parameterized [CVXPY](https://www.cvxpy.org/)
optimization problem and generates a Rust solver crate tailored to that problem
family. The generated crate reconstructs canonical cone-program data and solves
it with [Clarabel](https://clarabel.org/stable/). It also includes a Python
wrapper that can be registered as a custom CVXPY solve method for prototyping.

## Installation

This package is currently in an early stage of development. The recommended
setup is a development install from this repository.

We use [uv](https://github.com/astral-sh/uv) to manage dependencies. Once `uv`
is installed, run:

```bash
make sync
```

This installs the default development environment defined by the repository
`Makefile`.

## Quick Start

Generate a small nonnegative least-squares solver as a Rust crate:

```python
import cvxpy as cp
import cvxgenrust as cgr

m, n = 3, 2
A = cp.Parameter((m, n), name="A")
b = cp.Parameter(m, name="b")
x = cp.Variable(n, name="x")

problem = cp.Problem(
    cp.Minimize(cp.sum_squares(A @ x - b)),
    [x >= 0],
)

project = cgr.generate_code(
    problem,
    code_dir="nonneg_ls_cgr",
    module_name="nonneg_ls",
)
print("generated:", project.output_dir)
```

You should always set `name=` on CVXPY parameters and variables. The generated Rust
setters, extractors, metadata, and Python wrapper use those names after code
generation.

An HTML summary of the generated project is written to
`nonneg_ls_cgr/README.html`.

You can build and run the generated Rust project with:

```bash
cd nonneg_ls_cgr
cargo run --example solve
```

## Related projects

- [CVXPYgen](https://github.com/cvxgrp/cvxpygen): C code generation from CVXPY
  problems.
- [CVXGEN](https://cvxgen.com/docs/index.html): C code generation for convex
  optimization in MATLAB.
