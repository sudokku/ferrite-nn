use std::sync::{Arc, Mutex, atomic::AtomicBool, mpsc};
use ferrite_nn::{Network, NetworkSpec, EpochStats};

// ---------------------------------------------------------------------------
// Hyperparams
// ---------------------------------------------------------------------------

/// Training hyperparameters kept separate from the NetworkSpec so that the
/// architecture can be saved/loaded independently of how it is trained.
#[derive(Debug, Clone)]
pub struct Hyperparams {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
}

impl Default for Hyperparams {
    fn default() -> Self {
        Hyperparams { learning_rate: 0.01, batch_size: 32, epochs: 50 }
    }
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

/// Loaded dataset split into train / validation sets.
#[derive(Debug, Clone)]
pub struct DatasetState {
    pub train_inputs: Vec<Vec<f64>>,
    pub train_labels: Vec<Vec<f64>>,
    pub val_inputs:   Vec<Vec<f64>>,
    pub val_labels:   Vec<Vec<f64>>,
    pub feature_count: usize,
    pub label_count:   usize,
    pub total_rows:    usize,
    pub val_split_pct: u8,
    /// Short name displayed in the UI (e.g. "XOR", "circles", or file name stem).
    pub source_name:   String,
    /// First 5 rows of raw input for the preview table (inputs + labels).
    pub preview_rows:  Vec<(Vec<f64>, Vec<f64>)>,
}

// ---------------------------------------------------------------------------
// Training status
// ---------------------------------------------------------------------------

pub enum TrainingStatus {
    /// No training has been started yet.
    Idle,
    /// Training is running in a background thread.
    Running {
        stop_flag:    Arc<AtomicBool>,
        epoch_rx:     Arc<Mutex<mpsc::Receiver<EpochStats>>>,
        total_epochs: usize,
    },
    /// Training completed (naturally or via Stop) and the model was saved.
    /// `was_stopped` is true when the user clicked Stop before all epochs finished.
    Done {
        model_path:       String,
        elapsed_total_ms: u64,
        was_stopped:      bool,
    },
    /// Training failed with an error.
    Failed {
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Flash messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum FlashKind { Success, Error }

#[derive(Debug, Clone)]
pub struct FlashMessage {
    pub kind: FlashKind,
    pub text: String,
}

impl FlashMessage {
    pub fn success(text: impl Into<String>) -> Self {
        FlashMessage { kind: FlashKind::Success, text: text.into() }
    }
    pub fn error(text: impl Into<String>) -> Self {
        FlashMessage { kind: FlashKind::Error, text: text.into() }
    }
}

// ---------------------------------------------------------------------------
// Main state struct
// ---------------------------------------------------------------------------

pub struct StudioState {
    /// Saved architecture + loss type.
    pub spec:             Option<NetworkSpec>,
    /// Training hyperparameters.
    pub hyperparams:      Option<Hyperparams>,
    /// Loaded dataset.
    pub dataset:          Option<DatasetState>,
    /// Current training lifecycle state.
    pub training:         TrainingStatus,
    /// History of all epoch stats from the most recent training run.
    pub epoch_history:    Vec<EpochStats>,
    /// The trained network (available after training completes).
    pub trained_network:  Option<Network>,
    /// One-shot flash message for the next page render.
    pub flash:            Option<FlashMessage>,
}

impl StudioState {
    pub fn new() -> Self {
        StudioState {
            spec:            None,
            hyperparams:     None,
            dataset:         None,
            training:        TrainingStatus::Idle,
            epoch_history:   Vec::new(),
            trained_network: None,
            flash:           None,
        }
    }

    /// Returns a bitmask encoding which tabs should be unlocked.
    ///
    /// Bit layout:
    /// - bit 0 (Architect) — always set
    /// - bit 1 (Dataset)   — spec is saved
    /// - bit 2 (Train)     — dataset is loaded
    /// - bit 3 (Evaluate)  — training is Done or Stopped
    /// - bit 4 (Test)      — always set
    pub fn tab_unlock_mask(&self) -> u8 {
        let mut mask: u8 = 0b0_0001; // Architect always unlocked
        mask |= 0b1_0000; // Test always unlocked

        if self.spec.is_some() {
            mask |= 0b0_0010; // Dataset
        }
        if self.dataset.is_some() {
            mask |= 0b0_0100; // Train
        }
        match &self.training {
            TrainingStatus::Done { .. } => {
                mask |= 0b0_1000; // Evaluate
            }
            _ => {}
        }
        mask
    }

    /// Takes and returns the current flash message, clearing it.
    pub fn take_flash(&mut self) -> Option<FlashMessage> {
        self.flash.take()
    }
}

/// Shared state type — an `Arc<Mutex<StudioState>>` passed to every handler.
pub type SharedState = Arc<Mutex<StudioState>>;
