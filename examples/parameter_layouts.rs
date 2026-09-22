use parameter_layouts_cgr::{CGRProblem, ParameterLayout, SparseParameterPattern};

const EXPECTED_X: [f64; 3] = [0.4, 0.7, 0.2];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut problem = CGRProblem::new();

    let m = problem
        .parameter_info()
        .iter()
        .find(|parameter| parameter.name == "M")
        .expect("missing M metadata");
    assert_eq!(m.shape, &[3, 3]);
    assert_eq!(m.size, 9);
    assert!(matches!(m.layout, ParameterLayout::DenseColumnMajor));

    let s = problem
        .parameter_info()
        .iter()
        .find(|parameter| parameter.name == "S")
        .expect("missing S metadata");
    assert_eq!(s.shape, &[3, 3]);
    assert_eq!(s.size, 3);
    match s.layout {
        ParameterLayout::Sparse(SparseParameterPattern::Explicit { flat_indices }) => {
            assert_eq!(flat_indices, &[6, 1, 5]);
        }
        other => panic!("unexpected S layout: {other:?}"),
    }

    // Dense matrices use column-major order: the first three values are the
    // first column, followed by the second and third columns.
    problem.set_m(&[
        2.0, 0.0, 0.3, // column 0
        0.1, 1.5, 0.0, // column 1
        0.0, 0.2, 1.25, // column 2
    ])?;

    // Sparse values follow S.sparse_idx order: (0, 2), (1, 0), (2, 1).
    problem.set_s(&[0.4, -0.2, 0.15])?;
    problem.set_b(&[0.95, 1.01, 0.475])?;

    let solution = problem.solve()?;
    let x = problem.extract_x(&solution.x)?;
    for (actual, expected) in x.iter().zip(EXPECTED_X) {
        assert!(
            (*actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    println!("status = {}", solution.status);
    println!("objective = {}", solution.obj_val);
    println!("x = {x:?}");

    Ok(())
}
