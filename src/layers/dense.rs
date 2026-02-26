use crate::{math::matrix::Matrix, activation::activation::ActivationFunction};

#[derive(Debug)]
pub struct Layer{
    pub size: usize,
    pub neurons: Matrix,
    pre_neurons: Matrix,  // pre-activation values (z = Wx + b) needed for correct derivative
    pub weights: Matrix,
    pub biases: Matrix,
    pub activator: ActivationFunction
}

impl Layer {
    pub fn new(size: usize, input_size: usize, activation: ActivationFunction) -> Layer {
        let neurons = Matrix::random(1, size);
        let pre_neurons = Matrix::zeros(1, size);
        let weights = Matrix::random(input_size, size);
        let biases = Matrix::random(1, size);

        Layer {
            size,
            neurons,
            pre_neurons,
            weights,
            biases,
            activator: activation
        }
    }

    pub fn feed_from(&mut self, input: Vec<f64>) -> Vec<f64> {
        let z = Matrix::from_data(vec![input]) * self.weights.clone() + self.biases.clone();
        let a = z.map(|x| self.activator.function(x));
        self.pre_neurons = z;
        self.neurons = a.clone();
        a.data[0].clone()
    }

    /// Computes gradient adjustments. Returns (weights_grad, biases_grad).
    /// `next_layer_delta` is ∂L/∂a for this layer (error in activation space).
    pub fn compute_gradients(
        &self,
        next_layer_delta: Matrix,
        inputs: &Matrix,
    ) -> (Matrix, Matrix) {
        // Use pre-activation z so that derivative(z) = σ'(z) is computed correctly
        let act_derivative = self.pre_neurons.map(|x| self.activator.derivative(x));
        // Element-wise (Hadamard) product: δ = error ⊙ σ'(z)
        let layer_delta = hadamard(&next_layer_delta, &act_derivative);

        let weights_adjustment = inputs.transpose() * layer_delta.clone();
        let biases_adjustment = layer_delta;

        (weights_adjustment, biases_adjustment)
    }

    /// Applies pre-computed gradients scaled by lr.
    pub fn apply_gradients(&mut self, weights_grad: Matrix, biases_grad: Matrix, lr: f64) {
        self.weights = self.weights.clone() - weights_grad.map(|x| x * lr);
        self.biases = self.biases.clone() - biases_grad.map(|x| x * lr);
    }
}

/// Element-wise (Hadamard) product of two same-shape matrices.
fn hadamard(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let data = a.data.iter().zip(b.data.iter())
        .map(|(row_a, row_b)| {
            row_a.iter().zip(row_b.iter()).map(|(x, y)| x * y).collect()
        })
        .collect();
    Matrix::from_data(data)
}
