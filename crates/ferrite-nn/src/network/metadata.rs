use serde::{Deserialize, Serialize};

/// Describes how to interpret the input fed to a Network.
/// Stored in model JSON; GUI reads this to render the right input widget.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputType {
    /// Comma-separated f64 values — always valid fallback.
    Numeric,
    /// Grayscale image resized to width×height, normalized to [0, 1].
    ImageGrayscale { width: u32, height: u32 },
    /// RGB image resized to width×height, normalized to [0, 1], flattened as R,G,B,...
    ImageRgb { width: u32, height: u32 },
}

/// Optional annotations attached to a saved Network.
/// All fields are Option<> so old models (without metadata) deserialize cleanly.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMetadata {
    pub description: Option<String>,
    pub input_type: Option<InputType>,
    /// Human-readable class labels for the output layer (e.g. ["0","1",...,"9"]).
    pub output_labels: Option<Vec<String>>,
}
