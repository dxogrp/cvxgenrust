# CvxGenRust

`cvxgenrust` generates a Rust crate from a parameterized CVXPY problem.
The generated crate reconstructs [Clarabel](https://clarabel.org/stable/) canonical data and can be used from both Rust and Python.

## Installation

This package is currently in an early stage of development, so the recommended setup is a development install.

We use [uv](https://github.com/astral-sh/uv) to manage dependencies.
Once you have installed uv you can perform a development install with:

```bash
make sync
```

This installs the default development environment defined by the repository `Makefile`.

## Usage

Generate a small nonnegative least-squares solver as a Rust crate:

```python
import cvxpy as cp
import cvxgenrust as cgr

m, n = 3, 2
x = cp.Variable(n, name="x")
A = cp.Parameter((m, n), name="A")
b = cp.Parameter(m, name="b")

prob = cp.Problem(
    cp.Minimize(cp.sum_squares(A @ x - b)),
    [x >= 0],
)

project = cgr.generate_code(
    prob,
    code_dir="nnls_cgr",
    module_name="nnls",
)
print("generated:", project.output_dir)
```

Then build and run the generated Rust example:

```bash
cd nnls_cgr
cargo run --example solve
```

## Related projects

* [CVXPYgen](https://github.com/cvxgrp/cvxpygen): C code generation from CVXPY problems.
* [CVXGEN](https://cvxgen.com/docs/index.html): C code generation for convex optimization in MATLAB.
