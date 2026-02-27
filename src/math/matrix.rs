use rand::prelude::*;
use serde::{Serialize, Deserialize};
use std::f64::consts::PI;
use std::ops::{Add, Sub, Mul};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matrix{
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>
}

impl Matrix{
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix{
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows]
        }
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::thread_rng();
        let mut res = Matrix::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;
            }

        }

        res
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
        let mut res = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = Matrix::sample_standard_normal(&mut rng) * std_dev;
            }
        }
        res
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
        let mut res = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = Matrix::sample_standard_normal(&mut rng) * std_dev;
            }
        }
        res
    }

    pub fn transpose(&self) -> Matrix {
        let mut res = Matrix::zeros(self.cols, self.rows);

        for i in 0..res.rows {
            for j in 0..res.cols {
                res.data[i][j] = self.data[j][i];
            }
        }

        res
    }

    pub fn map<F>(&self, functor: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        Matrix::from_data(
            (self.data)
                .clone()
                .into_iter()
                .map(|row| row.into_iter().map(|x| functor(x)).collect())
                .collect()
        )
    }

    pub fn from_data(data: Vec<Vec<f64>>) -> Matrix {
        Matrix {
            rows: data.len(),
            cols: data[0].len(),
            data
        }
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Matrix { rows: 0, cols: 0, data: vec![] }
    }
}

impl Add for Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            panic!("Matrices are of incorrect sizes")
        }

        let mut res = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }

        res
    }
}

impl Sub for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            panic!("Matrices are of incorrect sizes")
        }

        let mut res = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }

        res
    }
}

impl Mul for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.cols != rhs.rows {
            panic!("Matrices are of incorrect sizes")
        }

        let mut res =  Matrix::zeros(self.rows, rhs.cols);

        for i in 0..res.rows {
            for j in 0..res.cols {
                let mut sum = 0.0;

                for k in 0..self.cols {
                    sum += self.data[i][k] * rhs.data[k][j];
                }

                res.data[i][j] = sum;
            }
        }

        res
    }
}
