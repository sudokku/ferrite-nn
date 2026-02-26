use crate::{
    math::matrix::Matrix,
    network::network::Network,
    loss::mse::MseLoss,
    optim::sgd::Sgd,
};

pub fn train_network(
    network: &mut Network,
    inputs: &[Vec<f64>],
    expected_outputs: &[Vec<f64>],
    optimizer: &Sgd,
) -> f64 {
    let mut total_loss = 0.0;

    for (input, expected) in inputs.iter().zip(expected_outputs.iter()) {
        // Forward pass
        let output = network.forward(input.clone());

        // Accumulate loss
        total_loss += MseLoss::loss(&output, expected);

        // Initial delta: ∂L/∂a_output (error in output activation space)
        let error = MseLoss::derivative(&output, expected);
        let mut delta = Matrix::from_data(vec![error]);

        // Backward pass
        for i in (0..network.layers.len()).rev() {
            let input_for_layer = if i == 0 {
                Matrix::from_data(vec![input.clone()])
            } else {
                network.layers[i - 1].neurons.clone()
            };

            // Borrow-checker ordering: compute gradients → compute next delta → apply step
            let (w_grad, b_grad) = network.layers[i].compute_gradients(delta.clone(), &input_for_layer);

            if i > 0 {
                // Propagate δ_i through weights to get ∂L/∂a_{i-1}
                delta = b_grad.clone() * network.layers[i].weights.transpose();
            }

            optimizer.step(&mut network.layers[i], w_grad, b_grad);
        }
    }

    total_loss / inputs.len() as f64
}
