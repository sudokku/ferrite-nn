/// Categorical cross-entropy loss for use with a Softmax output layer.
pub struct CrossEntropyLoss;

/// Small epsilon added inside log() to prevent log(0) = -inf.
const EPS: f64 = 1e-12;

impl CrossEntropyLoss {
    /// Computes the scalar cross-entropy loss:
    ///   L = -sum(expected[i] * log(predicted[i] + eps))
    ///
    /// `predicted` — softmax probabilities, shape [n_classes]
    /// `expected`  — one-hot (or soft) target distribution, shape [n_classes]
    pub fn loss(predicted: &[f64], expected: &[f64]) -> f64 {
        predicted.iter().zip(expected.iter())
            .map(|(p, e)| -e * (p + EPS).ln())
            .sum()
    }

    /// Gradient of the combined Softmax + cross-entropy w.r.t. the pre-softmax
    /// logits (i.e. the inputs to the Softmax layer).
    ///
    /// When Softmax and cross-entropy are composed together the gradient
    /// simplifies to:
    ///   ∂L/∂z_i = predicted[i] - expected[i]   (element-wise)
    ///
    /// This is the initial delta passed into the backward pass by the trainer.
    /// The Softmax layer's own derivative step should then be identity (1.0)
    /// so the combined gradient is not double-applied.
    pub fn derivative(predicted: &[f64], expected: &[f64]) -> Vec<f64> {
        predicted.iter().zip(expected.iter())
            .map(|(p, e)| p - e)
            .collect()
    }
}
