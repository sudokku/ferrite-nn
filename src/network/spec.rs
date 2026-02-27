use serde::{Serialize, Deserialize};
use crate::activation::activation::ActivationFunction;
use crate::loss::loss_type::LossType;
use crate::network::metadata::ModelMetadata;

/// Describes one layer in a network specification.
///
/// Fields:
/// - `size`       — number of neurons in this layer
/// - `input_size` — number of neurons feeding into this layer (i.e. the output
///                  size of the previous layer, or the raw input dimension for
///                  the first layer)
/// - `activation` — activation function applied after the linear transform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    pub size: usize,
    pub input_size: usize,
    pub activation: ActivationFunction,
}

/// A fully serializable description of a network architecture plus its
/// training loss type and optional metadata.
///
/// `NetworkSpec` can be saved to / loaded from JSON independently of the
/// trained weights, making it possible to store architecture configurations
/// before training starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSpec {
    /// Human-readable name used as the model file stem.
    pub name: String,
    /// Ordered list of layer descriptions (input → output).
    pub layers: Vec<LayerSpec>,
    /// Loss function to pair with this network during training.
    pub loss: LossType,
    /// Optional metadata (description, input type, output labels).
    #[serde(default)]
    pub metadata: Option<ModelMetadata>,
}

impl NetworkSpec {
    /// Serializes the spec to a pretty-printed JSON file.
    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Deserializes a `NetworkSpec` from a JSON file.
    pub fn load_json(path: &str) -> std::io::Result<NetworkSpec> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
