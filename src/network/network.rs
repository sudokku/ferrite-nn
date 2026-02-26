use crate::{activation::activation::ActivationFunction, layers::dense::Layer};

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
}
