use crate::activation::activation::ActivationFunction;
use crate::layers::dense::{ForwardCache, Layer};
use crate::network::metadata::ModelMetadata;
use crate::network::spec::NetworkSpec;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<Layer>,
    #[serde(default)]
    pub metadata: Option<ModelMetadata>,
}

impl Network {
    /// Builds a network from `(size, input_size, activation)` tuples.
    pub fn new(layer_specs: Vec<(usize, usize, ActivationFunction)>) -> Network {
        let layers = layer_specs
            .into_iter()
            .map(|(size, input_size, activation)| Layer::new(size, input_size, activation))
            .collect();
        Network {
            layers,
            metadata: None,
        }
    }

    /// Inference-only forward pass.
    ///
    /// Takes `&self` (not `&mut self`), so `Network` is `Sync` and an
    /// `Arc<Network>` can be shared safely across threads without cloning.
    /// Layer caches are discarded; use `forward_with_cache` during training.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current: Vec<f64> = input.to_vec();
        for layer in &self.layers {
            let (output, _cache) = layer.feed_from(&current);
            current = output;
        }
        current
    }

    /// Training forward pass — returns the final output **and** per-layer caches.
    ///
    /// The returned `ForwardCache` contains one `LayerCache` per layer (in
    /// forward order) with pre- and post-activation values needed by backprop.
    pub fn forward_with_cache(&self, input: &[f64]) -> (Vec<f64>, ForwardCache) {
        let mut current: Vec<f64> = input.to_vec();
        let mut cache: ForwardCache = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let (output, layer_cache) = layer.feed_from(&current);
            cache.push(layer_cache);
            current = output;
        }

        (current, cache)
    }

    /// Serializes the network weights to a pretty-printed JSON file.
    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Deserializes a network from a JSON file previously written by `save_json`.
    pub fn load_json(path: &str) -> std::io::Result<Network> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Returns the number of input neurons the network expects.
    /// This is the row count of the first layer's weight matrix (= fan-in).
    pub fn input_size(&self) -> usize {
        self.layers.first().map(|l| l.weights.rows).unwrap_or(0)
    }

    /// Returns a reference to the output layer's activation function.
    pub fn output_activation(&self) -> Option<&ActivationFunction> {
        self.layers.last().map(|l| &l.activator)
    }

    /// Builds a fresh (randomly initialized) `Network` from a `NetworkSpec`.
    ///
    /// Weight initialization follows `Layer::new` conventions:
    /// - ReLU activations → He init
    /// - everything else  → Xavier init
    ///
    /// Metadata is copied from the spec if present.
    pub fn from_spec(spec: &NetworkSpec) -> Network {
        let layers = spec
            .layers
            .iter()
            .map(|ls| Layer::new(ls.size, ls.input_size, ls.activation.clone()))
            .collect();
        Network {
            layers,
            metadata: spec.metadata.clone(),
        }
    }
}
