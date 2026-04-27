__HEADER__

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
}

impl CsrMatrix {
    pub fn mul_dense_vec(&self, rhs: &[f64]) -> Result<Vec<f64>, RuntimeError> {
        if rhs.len() != self.cols {
            return Err(RuntimeError::DimensionMismatch {
                expected: self.cols,
                actual: rhs.len(),
                name: "csr_rhs",
            });
        }

        let mut out = vec![0.0; self.rows];
        for row in 0..self.rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            let mut acc = 0.0;
            for slot in start..end {
                acc += self.data[slot] * rhs[self.indices[slot]];
            }
            out[row] = acc;
        }
        Ok(out)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CscMatrix {
    pub rows: usize,
    pub cols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
}

impl CscMatrix {
    pub fn to_dense_rows(&self) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.0; self.cols]; self.rows];
        for col in 0..self.cols {
            for slot in self.indptr[col]..self.indptr[col + 1] {
                out[self.indices[slot]][col] = self.data[slot];
            }
        }
        out
    }

    pub fn scaled(&self, alpha: f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: self.data.iter().map(|value| alpha * *value).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatrixPattern {
    pub rows: usize,
    pub cols: usize,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AffineCscMap {
    pub reduced: CsrMatrix,
    pub pattern: MatrixPattern,
}

impl AffineCscMap {
    pub fn apply(&self, parameter_vector: &[f64]) -> Result<CscMatrix, RuntimeError> {
        let values = self.reduced.mul_dense_vec(parameter_vector)?;
        Ok(CscMatrix {
            rows: self.pattern.rows,
            cols: self.pattern.cols,
            indptr: self.pattern.indptr.clone(),
            indices: self.pattern.indices.clone(),
            data: values,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AffineOffsetCscMap {
    pub reduced: CsrMatrix,
    pub pattern: MatrixPattern,
}

impl AffineOffsetCscMap {
    pub fn apply(
        &self,
        parameter_vector: &[f64],
    ) -> Result<(CscMatrix, Vec<f64>), RuntimeError> {
        let values = self.reduced.mul_dense_vec(parameter_vector)?;
        if values.len() != self.pattern.indices.len() {
            return Err(RuntimeError::MalformedMap(
                "offset CSC map output length does not match CSC pattern",
            ));
        }

        let a_cols = self.pattern.cols - 1;
        let a_nnz = self.pattern.indptr[a_cols];
        let mut offset = vec![0.0; self.pattern.rows];
        for slot in self.pattern.indptr[a_cols]..self.pattern.indptr[a_cols + 1] {
            offset[self.pattern.indices[slot]] = values[slot];
        }

        Ok((
            CscMatrix {
                rows: self.pattern.rows,
                cols: a_cols,
                indptr: self.pattern.indptr[..=a_cols].to_vec(),
                indices: self.pattern.indices[..a_nnz].to_vec(),
                data: values[..a_nnz].to_vec(),
            },
            offset,
        ))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AffineVectorMap {
    pub reduced: CsrMatrix,
    pub output_len: usize,
}

impl AffineVectorMap {
    pub fn apply(
        &self,
        parameter_vector: &[f64],
    ) -> Result<(Vec<f64>, f64), RuntimeError> {
        let out = self.reduced.mul_dense_vec(parameter_vector)?;
        if out.len() != self.output_len + 1 {
            return Err(RuntimeError::MalformedMap(
                "vector map output length does not match expected size",
            ));
        }
        let constant = *out
            .last()
            .ok_or(RuntimeError::MalformedMap("vector map has empty output"))?;
        Ok((out[..self.output_len].to_vec(), constant))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParameterInfo {
    pub name: &'static str,
    pub shape: &'static [usize],
    pub size: usize,
    pub offset: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VariableInfo {
    pub name: &'static str,
    pub shape: &'static [usize],
    pub size: usize,
    pub offset: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConeDims {
    pub zero: usize,
    pub nonneg: usize,
    pub exp: usize,
    pub soc: Vec<usize>,
    pub psd: Vec<usize>,
    pub p3d: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalConeQp {
    pub p: CscMatrix,
    pub c: Vec<f64>,
    pub objective_offset: f64,
    pub a: CscMatrix,
    pub b: Vec<f64>,
    pub cones: ConeDims,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolveResult {
    pub x: Vec<f64>,
    pub z: Vec<f64>,
    pub s: Vec<f64>,
    pub status: String,
    pub obj_val: f64,
    pub obj_val_dual: f64,
    pub solve_time: f64,
    pub iterations: u32,
    pub r_prim: f64,
    pub r_dual: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeError {
    UnknownParameter(String),
    UnknownVariable(String),
    Solver(String),
    DimensionMismatch {
        expected: usize,
        actual: usize,
        name: &'static str,
    },
    MalformedMap(&'static str),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::UnknownParameter(name) => write!(f, "unknown parameter `{name}`"),
            RuntimeError::UnknownVariable(name) => write!(f, "unknown variable `{name}`"),
            RuntimeError::Solver(message) => write!(f, "{message}"),
            RuntimeError::DimensionMismatch {
                expected,
                actual,
                name,
            } => write!(
                f,
                "dimension mismatch for {name}: expected {expected}, got {actual}"
            ),
            RuntimeError::MalformedMap(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for RuntimeError {}
