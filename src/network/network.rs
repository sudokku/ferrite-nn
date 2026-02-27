use crate::{activation::activation::ActivationFunction, layers::dense::Layer};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    /// Builds a network from (size, input_size, activation) tuples.
    pub fn new(layer_specs: Vec<(usize, usize, ActivationFunction)>) -> Network {
        let layers = layer_specs.into_iter()
            .map(|(size, input_size, activation)| Layer::new(size, input_size, activation))
            .collect();
        Network { layers }
    }

    /// Forward pass; stores activations in each layer for backprop.
    pub fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        let mut current = input;
        for layer in &mut self.layers {
            current = layer.feed_from(current);
        }
        current
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
}
