use crate::{activation::activation::ActivationFunction, math::matrix::Matrix};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ForwardCache — returned by feed_from / forward_with_cache
// ---------------------------------------------------------------------------

/// Per-layer cache produced by a single forward pass through one layer.
///
/// Stores both the pre-activation values (`z = W·x + b`) and the
/// post-activation values (`a = σ(z)`) so the backward pass can compute
/// `σ'(z)` and propagate deltas without touching mutable layer state.
#[derive(Debug, Clone)]
pub struct LayerCache {
    /// Pre-activation values `z = W·x + b`.  Shape: `[size]`.
    pub pre_activation: Vec<f64>,
    /// Post-activation values `a = σ(z)`.  Shape: `[size]`.
    pub post_activation: Vec<f64>,
}

/// One `LayerCache` entry per layer, ordered from input → output.
pub type ForwardCache = Vec<LayerCache>;

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// A single fully-connected layer.
///
/// # Weight layout
/// `weights` has shape `(input_size, size)` stored row-major.
/// Element `weights[k, j]` = connection from input neuron `k` to output neuron `j`.
///
/// `biases` has shape `(1, size)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub size: usize,
    pub weights: Matrix,
    pub biases: Matrix,
    pub activator: ActivationFunction,
}

impl Layer {
    /// Creates a new layer with the given size and activation function.
    ///
    /// Weight initialization:
    /// - `ReLU`  → He init   (variance = 2 / fan_in)
    /// - others  → Xavier init (variance = 1 / fan_in)
    ///
    /// Biases are initialized to zero.
    pub fn new(size: usize, input_size: usize, activation: ActivationFunction) -> Layer {
        let weights = match activation {
            ActivationFunction::ReLU => Matrix::he(input_size, size),
            _ => Matrix::xavier(input_size, size),
        };
        let biases = Matrix::zeros(1, size);

        Layer {
            size,
            weights,
            biases,
            activator: activation,
        }
    }

    /// Forward pass through this layer.
    ///
    /// Takes `input` as a slice of length `input_size` (= `self.weights.rows`).
    /// Returns `(output, cache)` where `output` is the post-activation vector of
    /// length `self.size` and `cache` holds `pre_activation` (z) and
    /// `post_activation` (a) for use during backpropagation.
    ///
    /// This method takes `&self` (not `&mut self`), making `Network` `Sync` and
    /// enabling safe parallel inference via `Arc<Network>`.
    pub fn feed_from(&self, input: &[f64]) -> (Vec<f64>, LayerCache) {
        let size = self.size;

        // ── z = W·x + b ─────────────────────────────────────────────────────
        // Inline zero-allocation matmul: weights is (input_size, size) row-major.
        // output[j] = Σ_k  input[k] * weights[k, j]  + biases[j]
        // Loop order: k then j — keeps weights row k contiguous in cache.
        let mut z = vec![0.0_f64; size];
        for k in 0..input.len() {
            let w_row = &self.weights.data[k * size..(k + 1) * size];
            let x_k = input[k];
            for j in 0..size {
                z[j] += x_k * w_row[j];
            }
        }
        for j in 0..size {
            z[j] += self.biases.data[j];
        }

        // ── a = σ(z) ────────────────────────────────────────────────────────
        let a: Vec<f64> = match &self.activator {
            ActivationFunction::Softmax => {
                // Numerically stable softmax: subtract max(z) before exp to
                // prevent overflow while preserving the output distribution.
                let max_z = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = z.iter().map(|&v| (v - max_z).exp()).collect();
                let sum_exps: f64 = exps.iter().sum();
                exps.iter().map(|&e| e / sum_exps).collect()
            }
            _ => z.iter().map(|&v| self.activator.function(v)).collect(),
        };

        let cache = LayerCache {
            pre_activation: z,
            post_activation: a.clone(),
        };

        (a, cache)
    }

    /// Computes weight and bias gradients for this layer given the upstream delta
    /// and the explicit pre-activation cache.
    ///
    /// # Arguments
    /// - `next_layer_delta` — ∂L/∂a for this layer (error in activation space),
    ///   shape `(1, size)`.
    /// - `inputs`           — post-activation output of the **previous** layer (or
    ///   the raw network input for layer 0), shape `(1, input_size)`.
    /// - `pre_activation`   — `z` values cached during `feed_from`, length `size`.
    ///
    /// # Returns
    /// `(weights_grad, biases_grad)` — raw (unscaled) gradient matrices.
    pub fn compute_gradients(
        &self,
        next_layer_delta: Matrix,
        inputs: &Matrix,
        pre_activation: &[f64],
    ) -> (Matrix, Matrix) {
        // σ'(z) applied element-wise
        let act_derivative = Matrix {
            rows: 1,
            cols: self.size,
            data: pre_activation
                .iter()
                .map(|&z| self.activator.derivative(z))
                .collect(),
        };

        // Element-wise (Hadamard) product: δ = error ⊙ σ'(z)
        let layer_delta = hadamard(&next_layer_delta, &act_derivative);

        let weights_adjustment = inputs.transpose() * layer_delta.clone();
        let biases_adjustment = layer_delta;

        (weights_adjustment, biases_adjustment)
    }

    /// Applies pre-computed gradients in-place using SGD: `w -= lr * grad`.
    ///
    /// Uses `SubAssign` to avoid an extra clone of `self.weights`/`self.biases`.
    pub fn apply_gradients(&mut self, weights_grad: Matrix, biases_grad: Matrix, lr: f64) {
        self.weights -= weights_grad.map(|x| x * lr);
        self.biases -= biases_grad.map(|x| x * lr);
    }
}

// ---------------------------------------------------------------------------
// Hadamard (element-wise) product
// ---------------------------------------------------------------------------

/// Element-wise (Hadamard) product of two same-shape matrices.
pub(crate) fn hadamard(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.rows, b.rows, "hadamard: row mismatch");
    assert_eq!(a.cols, b.cols, "hadamard: col mismatch");
    let data: Vec<f64> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).collect();
    Matrix {
        rows: a.rows,
        cols: a.cols,
        data,
    }
}
