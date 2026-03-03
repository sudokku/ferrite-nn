use std::sync::mpsc;
use std::sync::{Arc, atomic::AtomicBool};
use crate::loss::loss_type::LossType;
use crate::train::epoch_stats::EpochStats;

/// Configuration for a `train_loop` run.
///
/// # Fields
/// - `epochs`      — total number of full passes over the training data
/// - `batch_size`  — samples per mini-batch; use `1` for online SGD
/// - `loss_type`   — which loss function to use (`Mse` or `CrossEntropy`)
/// - `progress_tx` — optional channel sender; one `EpochStats` is sent per
///                   completed epoch.  If the receiver is dropped the loop
///                   terminates early (clean shutdown).
/// - `stop_flag`   — optional atomic flag; when set to `true` from another
///                   thread the loop terminates after the current epoch.
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub loss_type: LossType,
    pub progress_tx: Option<mpsc::Sender<EpochStats>>,
    pub stop_flag: Option<Arc<AtomicBool>>,
}

impl TrainConfig {
    /// Creates a minimal `TrainConfig` with no progress channel and no stop flag.
    pub fn new(epochs: usize, batch_size: usize, loss_type: LossType) -> Self {
        TrainConfig {
            epochs,
            batch_size,
            loss_type,
            progress_tx: None,
            stop_flag: None,
        }
    }
}
