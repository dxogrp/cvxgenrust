# CvxGenRust

`cvxgenrust` generates a Rust crate from a parameterized CVXPY problem.
The generated crate reconstructs [Clarabel](https://clarabel.org/stable/) canonical data and can be used from both Rust and Python.

## Usage

Generate a small nonnegative least-squares solver as a Rust crate:

```python
import cvxpy as cp
import cvxgenrust

m, n = 3, 2
x = cp.Variable(n, name="x")
A = cp.Parameter((m, n), name="A")
b = cp.Parameter(m, name="b")

problem = cp.Problem(
    cp.Minimize(cp.sum_squares(A @ x - b)),
    [x >= 0],
)

project = cvxgenrust.generate_code(
    problem,
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
