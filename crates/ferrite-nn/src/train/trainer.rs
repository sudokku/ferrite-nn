use rand::seq::SliceRandom;
use crate::{
    math::matrix::Matrix,
    network::network::Network,
    loss::mse::MseLoss,
    optim::sgd::Sgd,
};

/// Trains the network for one epoch using mini-batch SGD.
///
/// # Arguments
/// * `network`          — the network to train (mutated in place)
/// * `inputs`           — slice of input samples
/// * `expected_outputs` — corresponding target outputs, same length as `inputs`
/// * `optimizer`        — SGD optimizer (holds learning rate)
/// * `batch_size`       — number of samples per mini-batch; pass `1` for
///                        online (sample-by-sample) SGD
///
/// # Returns
/// Mean loss over all samples in the epoch.
pub fn train_network(
    network: &mut Network,
    inputs: &[Vec<f64>],
    expected_outputs: &[Vec<f64>],
    optimizer: &Sgd,
    batch_size: usize,
) -> f64 {
    assert!(!inputs.is_empty(), "inputs must not be empty");
    assert_eq!(inputs.len(), expected_outputs.len(), "inputs and expected_outputs must have equal length");
    assert!(batch_size > 0, "batch_size must be at least 1");

    let n = inputs.len();
    let mut total_loss = 0.0;

    // Shuffle indices so each epoch sees data in a different order.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rand::thread_rng());

    // Process in mini-batches.
    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let actual_batch_size = (batch_end - batch_start) as f64;

        // Initialize accumulated gradient storage: one (w_grad, b_grad) pair
        // per layer, all zeros with the correct shapes.
        let mut acc_grads: Vec<(Matrix, Matrix)> = network.layers.iter()
            .map(|layer| {
                (
                    Matrix::zeros(layer.weights.rows, layer.weights.cols),
                    Matrix::zeros(layer.biases.rows, layer.biases.cols),
                )
            })
            .collect();

        // Accumulate gradients over all samples in the mini-batch.
        for &idx in &indices[batch_start..batch_end] {
            let input = &inputs[idx];
            let expected = &expected_outputs[idx];

            // Forward pass — stores activations in each layer.
            let output = network.forward(input.clone());

            // Accumulate loss (for reporting).
            total_loss += MseLoss::loss(&output, expected);

            // Initial delta: ∂L/∂a_output
            let error = MseLoss::derivative(&output, expected);
            let mut delta = Matrix::from_data(vec![error]);

            // Backward pass — accumulate raw gradients (not yet scaled by lr).
            for i in (0..network.layers.len()).rev() {
                let input_for_layer = if i == 0 {
                    Matrix::from_data(vec![input.clone()])
                } else {
                    network.layers[i - 1].neurons.clone()
                };

                let (w_grad, b_grad) = network.layers[i].compute_gradients(
                    delta.clone(),
                    &input_for_layer,
                );

                if i > 0 {
                    // Propagate δ back through weights to the previous layer.
                    delta = b_grad.clone() * network.layers[i].weights.transpose();
                }

                // Accumulate: acc += grad  (element-wise addition)
                acc_grads[i].0 = acc_grads[i].0.clone() + w_grad;
                acc_grads[i].1 = acc_grads[i].1.clone() + b_grad;
            }
        }

        // Apply averaged gradients: divide accumulated sum by batch size, then
        // call the optimizer once per layer.
        let inv_batch = 1.0 / actual_batch_size;
        for (i, (w_acc, b_acc)) in acc_grads.into_iter().enumerate() {
            let w_avg = w_acc.map(|x| x * inv_batch);
            let b_avg = b_acc.map(|x| x * inv_batch);
            optimizer.step(&mut network.layers[i], w_avg, b_avg);
        }
    }

    total_loss / n as f64
}
