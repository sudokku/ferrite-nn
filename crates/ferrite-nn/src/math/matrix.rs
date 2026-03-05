use rand::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

/// A 2-D matrix stored in row-major flat `Vec<f64>`.
///
/// Element `(i, j)` (0-indexed) lives at `data[i * cols + j]`.
///
/// # Serialization
/// Serialized and deserialized as `Vec<Vec<f64>>` (nested row arrays) so that
/// JSON model files written by previous versions of Ferrite remain loadable.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    /// Flat row-major storage.  Length is always `rows * cols`.
    pub data: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl Matrix {
    /// Creates a zero-filled matrix of shape `(rows, cols)`.
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Fills a `(rows, cols)` matrix with uniform random values in `[-1, 1]`.
    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.gen::<f64>() * 2.0 - 1.0)
            .collect();
        Matrix { rows, cols, data }
    }

    /// Samples a single value from N(0, 1) using the Box-Muller transform.
    /// Both u1 and u2 must be uniform on (0, 1].
    fn sample_standard_normal(rng: &mut ThreadRng) -> f64 {
        // Draw two independent uniform samples in (0, 1] to avoid log(0).
        let u1: f64 = 1.0 - rng.gen::<f64>();
        let u2: f64 = 1.0 - rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// He initialization: samples from N(0, sqrt(2 / cols)).
    ///
    /// Recommended before ReLU layers. The variance 2/fan_in accounts for
    /// the fact that ReLU zeroes half of its inputs on average.
    ///
    /// Shape: (rows, cols). `cols` is the fan-in (number of input connections).
    pub fn he(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / cols as f64).sqrt();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| Matrix::sample_standard_normal(&mut rng) * std_dev)
            .collect();
        Matrix { rows, cols, data }
    }

    /// Xavier (Glorot) initialization: samples from N(0, sqrt(1 / cols)).
    ///
    /// Recommended before Sigmoid/Tanh/Identity layers. Keeps the variance of
    /// activations and gradients roughly equal across layers.
    ///
    /// Shape: (rows, cols). `cols` is the fan-in (number of input connections).
    pub fn xavier(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::thread_rng();
        let std_dev = (1.0 / cols as f64).sqrt();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| Matrix::sample_standard_normal(&mut rng) * std_dev)
            .collect();
        Matrix { rows, cols, data }
    }

    /// Transposes the matrix, returning a new `(cols, rows)` matrix.
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                // result[j, i] = self[i, j]
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        result
    }

    /// Applies `f` element-wise, returning a new matrix of the same shape.
    pub fn map<F>(&self, f: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Constructs a `Matrix` from a nested `Vec<Vec<f64>>`.
    ///
    /// The outer Vec is rows; each inner Vec must have the same length (cols).
    /// Panics if `data` is empty or rows have inconsistent lengths.
    pub fn from_data(nested: Vec<Vec<f64>>) -> Matrix {
        let rows = nested.len();
        assert!(rows > 0, "Matrix::from_data: data must not be empty");
        let cols = nested[0].len();
        let data: Vec<f64> = nested.into_iter().flatten().collect();
        assert_eq!(
            data.len(),
            rows * cols,
            "Matrix::from_data: all rows must have the same length"
        );
        Matrix { rows, cols, data }
    }

    /// Inline (non-allocating) matrix multiply: `lhs * rhs` returning a new matrix.
    ///
    /// Uses the `i, k, j` loop order to keep `lhs[i, k]` in a register while
    /// streaming through the contiguous `rhs` row `k`, improving cache behaviour
    /// compared to the naive `i, j, k` order.
    ///
    /// Panics if `lhs.cols != rhs.rows`.
    pub fn matmul(lhs: &Matrix, rhs: &Matrix) -> Matrix {
        assert_eq!(
            lhs.cols, rhs.rows,
            "Matrix::matmul: lhs.cols ({}) != rhs.rows ({})",
            lhs.cols, rhs.rows
        );
        let mut result = Matrix::zeros(lhs.rows, rhs.cols);
        for i in 0..lhs.rows {
            for k in 0..lhs.cols {
                let a_ik = lhs.data[i * lhs.cols + k];
                let rhs_row_start = k * rhs.cols;
                let res_row_start = i * rhs.cols;
                for j in 0..rhs.cols {
                    result.data[res_row_start + j] += a_ik * rhs.data[rhs_row_start + j];
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

impl Default for Matrix {
    fn default() -> Self {
        Matrix {
            rows: 0,
            cols: 0,
            data: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Arithmetic operators
// ---------------------------------------------------------------------------

impl Add for Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Matrix::add: row mismatch");
        assert_eq!(self.cols, rhs.cols, "Matrix::add: col mismatch");
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl AddAssign for Matrix {
    /// In-place element-wise addition.  Avoids the clone inherent in `a = a + b`.
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows, rhs.rows, "Matrix::add_assign: row mismatch");
        assert_eq!(self.cols, rhs.cols, "Matrix::add_assign: col mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a += b;
        }
    }
}

impl Sub for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows, "Matrix::sub: row mismatch");
        assert_eq!(self.cols, rhs.cols, "Matrix::sub: col mismatch");
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl SubAssign for Matrix {
    /// In-place element-wise subtraction.  Avoids the clone in `a = a - b`.
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows, rhs.rows, "Matrix::sub_assign: row mismatch");
        assert_eq!(self.cols, rhs.cols, "Matrix::sub_assign: col mismatch");
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a -= b;
        }
    }
}

impl Mul for Matrix {
    type Output = Matrix;

    /// Matrix multiplication via the `i, k, j` cache-friendly loop order.
    fn mul(self, rhs: Self) -> Self::Output {
        Matrix::matmul(&self, &rhs)
    }
}

// ---------------------------------------------------------------------------
// Serde — serialize/deserialize as nested Vec<Vec<f64>> for JSON compatibility
// ---------------------------------------------------------------------------

impl Serialize for Matrix {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Reconstruct row-slices on the fly; no extra allocation per row needed
        // since serde will consume the iterator immediately.
        let nested: Vec<&[f64]> = (0..self.rows)
            .map(|i| &self.data[i * self.cols..(i + 1) * self.cols])
            .collect();
        // We also need to persist rows/cols so that a zero-element matrix round-trips.
        // Encode as { "rows": N, "cols": M, "data": [[...], ...] }
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Matrix", 3)?;
        s.serialize_field("rows", &self.rows)?;
        s.serialize_field("cols", &self.cols)?;
        s.serialize_field("data", &nested)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for Matrix {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Accept both legacy format (object with rows/cols/data) and the same
        // format we now emit.  The "data" field is always Vec<Vec<f64>>.
        #[derive(Deserialize)]
        struct MatrixHelper {
            rows: usize,
            cols: usize,
            data: Vec<Vec<f64>>,
        }
        let h = MatrixHelper::deserialize(deserializer)?;
        let flat: Vec<f64> = h.data.into_iter().flatten().collect();
        Ok(Matrix {
            rows: h.rows,
            cols: h.cols,
            data: flat,
        })
    }
}
