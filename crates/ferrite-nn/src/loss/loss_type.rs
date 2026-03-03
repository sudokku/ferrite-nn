use serde::{Serialize, Deserialize};

/// Selects which loss function the training loop uses.
///
/// - `Mse`                — Mean-squared error; pair with Identity or Sigmoid output.
/// - `CrossEntropy`       — Categorical cross-entropy; pair with Softmax output.
///   The gradient is the combined Softmax+CE gradient (predicted - expected),
///   which matches the convention in `CrossEntropyLoss::derivative()`.
/// - `BinaryCrossEntropy` — Binary cross-entropy; pair with Sigmoid output.
/// - `Mae`                — Mean absolute error; pair with Identity output.
/// - `Huber`              — Huber loss (δ=1.0); pair with Identity output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossType {
    Mse,
    CrossEntropy,
    BinaryCrossEntropy,
    Mae,
    Huber,
}
