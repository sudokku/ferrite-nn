use serde::{Serialize, Deserialize};

/// Selects which loss function the training loop uses.
///
/// - `Mse`           — Mean-squared error; pair with Identity or Sigmoid output.
/// - `CrossEntropy`  — Categorical cross-entropy; pair with Softmax output.
///   The gradient is the combined Softmax+CE gradient (predicted - expected),
///   which matches the convention in `CrossEntropyLoss::derivative()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossType {
    Mse,
    CrossEntropy,
}
