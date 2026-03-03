use crate::{math::matrix::Matrix, layers::dense::Layer};

pub struct Sgd {
    pub learning_rate: f64,
}

impl Sgd {
    pub fn new(learning_rate: f64) -> Sgd {
        Sgd { learning_rate }
    }

    /// Applies one SGD weight update to a layer given its pre-computed gradients.
    pub fn step(&self, layer: &mut Layer, weights_grad: Matrix, biases_grad: Matrix) {
        layer.apply_gradients(weights_grad, biases_grad, self.learning_rate);
    }
}
