use std::sync::atomic::Ordering;
use std::time::Instant;

use rand::seq::SliceRandom;

use crate::loss::loss_type::LossType;
use crate::loss::mse::MseLoss;
use crate::loss::cross_entropy::CrossEntropyLoss;
use crate::math::matrix::Matrix;
use crate::network::network::Network;
use crate::optim::sgd::Sgd;
use crate::train::epoch_stats::EpochStats;
use crate::train::train_config::TrainConfig;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Trains `network` for `config.epochs` epochs and returns the mean training
/// loss of the **last completed epoch**.
///
/// # Arguments
/// - `network`      — mutable reference to the network; modified in place
/// - `train_inputs` — training samples, each a `Vec<f64>` of length `input_size`
/// - `train_labels` — corresponding targets, same length as `train_inputs`
/// - `val_inputs`   — optional validation samples
/// - `val_labels`   — optional validation targets (required iff `val_inputs` is `Some`)
/// - `optimizer`    — SGD optimizer (carries learning rate)
/// - `config`       — hyperparameters, optional progress channel, optional stop flag
///
/// # Early termination
/// The loop breaks early if:
/// - the `progress_tx` receiver has been dropped (natural SSE disconnect), **or**
/// - `config.stop_flag` is set to `true`.
///
/// # Panics
/// Panics if `train_inputs` is empty, lengths mismatch, or `batch_size == 0`.
pub fn train_loop(
    network: &mut Network,
    train_inputs: &[Vec<f64>],
    train_labels: &[Vec<f64>],
    val_inputs: Option<&[Vec<f64>]>,
    val_labels: Option<&[Vec<f64>]>,
    optimizer: &Sgd,
    config: &TrainConfig,
) -> f64 {
    assert!(!train_inputs.is_empty(), "train_inputs must not be empty");
    assert_eq!(
        train_inputs.len(),
        train_labels.len(),
        "train_inputs and train_labels must have equal length"
    );
    assert!(config.batch_size > 0, "batch_size must be at least 1");

    let mut last_train_loss = 0.0;

    for epoch in 1..=config.epochs {
        // Check stop flag at the top of each epoch.
        if let Some(ref flag) = config.stop_flag {
            if flag.load(Ordering::Relaxed) {
                break;
            }
        }

        let t_start = Instant::now();

        // ── One full pass over the training data ───────────────────────────
        let train_loss = run_one_epoch(
            network,
            train_inputs,
            train_labels,
            optimizer,
            config.batch_size,
            config.loss_type,
        );
        last_train_loss = train_loss;

        let elapsed_ms = t_start.elapsed().as_millis() as u64;

        // ── Accuracy (CrossEntropy only) ───────────────────────────────────
        let train_accuracy = if config.loss_type == LossType::CrossEntropy {
            Some(compute_accuracy(network, train_inputs, train_labels))
        } else {
            None
        };

        // ── Validation ────────────────────────────────────────────────────
        let (val_loss, val_accuracy) = if let (Some(vi), Some(vl)) = (val_inputs, val_labels) {
            let vl_val = compute_eval_loss(network, vi, vl, config.loss_type);
            let va = if config.loss_type == LossType::CrossEntropy {
                Some(compute_accuracy(network, vi, vl))
            } else {
                None
            };
            (Some(vl_val), va)
        } else {
            (None, None)
        };

        // ── Emit progress ─────────────────────────────────────────────────
        let stats = EpochStats {
            epoch,
            total_epochs: config.epochs,
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
            elapsed_ms,
        };

        if let Some(ref tx) = config.progress_tx {
            // If the receiver has been dropped, stop training.
            if tx.send(stats).is_err() {
                break;
            }
        }

        // Check stop flag again after potentially expensive eval.
        if let Some(ref flag) = config.stop_flag {
            if flag.load(Ordering::Relaxed) {
                break;
            }
        }
    }

    last_train_loss
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Runs one full epoch of mini-batch SGD over the training data.
/// Returns the mean loss over all samples.
fn run_one_epoch(
    network: &mut Network,
    inputs: &[Vec<f64>],
    labels: &[Vec<f64>],
    optimizer: &Sgd,
    batch_size: usize,
    loss_type: LossType,
) -> f64 {
    let n = inputs.len();
    let mut total_loss = 0.0;

    // Shuffle sample order each epoch.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rand::thread_rng());

    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let actual_batch_size = (batch_end - batch_start) as f64;

        // Zero-initialize accumulated gradient storage.
        let mut acc_grads: Vec<(Matrix, Matrix)> = network.layers.iter()
            .map(|layer| (
                Matrix::zeros(layer.weights.rows, layer.weights.cols),
                Matrix::zeros(layer.biases.rows, layer.biases.cols),
            ))
            .collect();

        // Accumulate gradients over the mini-batch.
        for &idx in &indices[batch_start..batch_end] {
            let input    = &inputs[idx];
            let expected = &labels[idx];

            let output = network.forward(input.clone());

            total_loss += compute_loss(&output, expected, loss_type);

            let error  = compute_loss_derivative(&output, expected, loss_type);
            let mut delta = Matrix::from_data(vec![error]);

            // Backward pass.
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
                    delta = b_grad.clone() * network.layers[i].weights.transpose();
                }

                acc_grads[i].0 = acc_grads[i].0.clone() + w_grad;
                acc_grads[i].1 = acc_grads[i].1.clone() + b_grad;
            }
        }

        // Average and apply.
        let inv_batch = 1.0 / actual_batch_size;
        for (i, (w_acc, b_acc)) in acc_grads.into_iter().enumerate() {
            let w_avg = w_acc.map(|x| x * inv_batch);
            let b_avg = b_acc.map(|x| x * inv_batch);
            optimizer.step(&mut network.layers[i], w_avg, b_avg);
        }
    }

    total_loss / n as f64
}

/// Scalar loss for one sample — dispatches on `LossType`.
fn compute_loss(predicted: &[f64], expected: &[f64], loss_type: LossType) -> f64 {
    match loss_type {
        LossType::Mse          => MseLoss::loss(predicted, expected),
        LossType::CrossEntropy => CrossEntropyLoss::loss(predicted, expected),
    }
}

/// Per-output gradient for one sample — dispatches on `LossType`.
fn compute_loss_derivative(predicted: &[f64], expected: &[f64], loss_type: LossType) -> Vec<f64> {
    match loss_type {
        LossType::Mse          => MseLoss::derivative(predicted, expected),
        LossType::CrossEntropy => CrossEntropyLoss::derivative(predicted, expected),
    }
}

/// Mean loss over a full dataset without gradient accumulation (eval mode).
fn compute_eval_loss(
    network: &mut Network,
    inputs: &[Vec<f64>],
    labels: &[Vec<f64>],
    loss_type: LossType,
) -> f64 {
    let n = inputs.len();
    if n == 0 {
        return 0.0;
    }
    let total: f64 = inputs.iter().zip(labels.iter())
        .map(|(input, label)| {
            let output = network.forward(input.clone());
            compute_loss(&output, label, loss_type)
        })
        .sum();
    total / n as f64
}

/// Fraction of samples classified correctly (argmax match).
/// Used for `CrossEntropy` runs only.
fn compute_accuracy(
    network: &mut Network,
    inputs: &[Vec<f64>],
    labels: &[Vec<f64>],
) -> f64 {
    let n = inputs.len();
    if n == 0 {
        return 0.0;
    }
    let correct: usize = inputs.iter().zip(labels.iter())
        .filter(|(input, label)| {
            let output = network.forward((*input).clone());
            argmax(&output) == argmax(label)
        })
        .count();
    correct as f64 / n as f64
}

/// Index of the maximum element in a slice.
fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
