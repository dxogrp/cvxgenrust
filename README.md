# cvxgenrust

`cvxgenrust` takes a parameterized [CVXPY](https://www.cvxpy.org/)
optimization problem and generates a Rust solver crate tailored to that problem
family. The generated crate reconstructs canonical cone-program data and solves
it with [Clarabel](https://clarabel.org/stable/). It also includes a Python
wrapper that can be registered as a custom CVXPY solve method for prototyping.

More details can be found in the associated [paper](https://haozhu10015.github.io/papers/cvxgenrust.html).

## Installation

Install the released package from PyPI with:

```bash
pip install cvxgenrust
```

Generated solver projects use Rust, Cargo, Clarabel, and, when the Python
wrapper is enabled, a PyO3/maturin build. Install a stable Rust toolchain before
building or importing generated extension wrappers.

For development from this repository, use [uv](https://github.com/astral-sh/uv)
to manage dependencies. Once `uv` is installed, run:


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

By default, `generate_code` also compiles the generated Python extension wrapper
into the generated project's `python/` directory. Pass `wrapper=False` to only
write the Rust crate and Python wrapper sources.

## Related projects

- [CVXPYgen](https://github.com/cvxgrp/cvxpygen): C code generation from CVXPY
  problems.
- [CVXGEN](https://cvxgen.com/docs/index.html): C code generation for convex
  optimization in MATLAB.
