use serde::{Serialize, Deserialize};

/// Per-epoch training statistics emitted by `train_loop`.
///
/// When a `progress_tx` channel is configured in `TrainConfig`, the training
/// loop sends one `EpochStats` value at the end of every completed epoch.
/// Receivers (e.g. the studio SSE handler) use this to drive real-time charts
/// and progress indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochStats {
    /// 1-based epoch number.
    pub epoch: usize,
    /// Total epochs requested for this run.
    pub total_epochs: usize,
    /// Mean training loss over all samples in this epoch.
    pub train_loss: f64,
    /// Mean validation loss, if a validation set was provided.
    pub val_loss: Option<f64>,
    /// Training accuracy as a fraction in [0, 1]; only set for CrossEntropy runs.
    pub train_accuracy: Option<f64>,
    /// Validation accuracy as a fraction in [0, 1]; only set for CrossEntropy runs
    /// when a validation set is available.
    pub val_accuracy: Option<f64>,
    /// Wall-clock duration of this single epoch in milliseconds.
    pub elapsed_ms: u64,
}
