/// MNIST digit classification example for ferrite-nn.
///
/// Architecture: 784 → 256 (ReLU) → 128 (ReLU) → 10 (Softmax)
/// Loss:         CrossEntropyLoss (combined with Softmax — gradient is predicted - expected)
/// Optimizer:    SGD, lr = 0.01
/// Batch size:   32
/// Epochs:       50
///
/// Run with:
///   cargo run --example mnist --release
///
/// Data files must be present at examples/mnist_data/ (IDX binary format).

use std::fs::File;
use std::io::{self, Read, Write};

use ferrite_nn::{
    Network,
    ActivationFunction,
    CrossEntropyLoss,
    Sgd,
    math::matrix::Matrix,
};
use rand::seq::SliceRandom;

// ---------------------------------------------------------------------------
// Data loading helpers
// ---------------------------------------------------------------------------

/// Reads an IDX3 image file and returns a Vec of 784-element f64 Vecs,
/// with pixel values normalized from [0, 255] to [0.0, 1.0].
fn load_images(path: &str) -> Vec<Vec<f64>> {
    let mut file = File::open(path)
        .unwrap_or_else(|e| panic!("Cannot open image file '{}': {}", path, e));

    // Parse header.
    let mut buf4 = [0u8; 4];

    file.read_exact(&mut buf4).expect("Failed to read magic number");
    let magic = i32::from_be_bytes(buf4);
    assert_eq!(magic, 0x00000803, "Image file magic number mismatch (got {:#010x})", magic);

    file.read_exact(&mut buf4).expect("Failed to read image count");
    let n_images = i32::from_be_bytes(buf4) as usize;

    file.read_exact(&mut buf4).expect("Failed to read row count");
    let rows = i32::from_be_bytes(buf4) as usize;

    file.read_exact(&mut buf4).expect("Failed to read col count");
    let cols = i32::from_be_bytes(buf4) as usize;

    let n_pixels = rows * cols;
    assert_eq!(n_pixels, 784, "Expected 28×28 images (784 pixels), got {}×{}={}", rows, cols, n_pixels);

    // Read all pixel bytes at once, then normalize.
    let mut pixel_bytes = vec![0u8; n_images * n_pixels];
    file.read_exact(&mut pixel_bytes).expect("Failed to read pixel data");

    pixel_bytes
        .chunks(n_pixels)
        .map(|chunk| chunk.iter().map(|&p| p as f64 / 255.0).collect())
        .collect()
}

/// Reads an IDX1 label file and returns a Vec of one-hot Vec<f64> of length 10.
fn load_labels(path: &str) -> Vec<Vec<f64>> {
    let mut file = File::open(path)
        .unwrap_or_else(|e| panic!("Cannot open label file '{}': {}", path, e));

    let mut buf4 = [0u8; 4];

    file.read_exact(&mut buf4).expect("Failed to read magic number");
    let magic = i32::from_be_bytes(buf4);
    assert_eq!(magic, 0x00000801, "Label file magic number mismatch (got {:#010x})", magic);

    file.read_exact(&mut buf4).expect("Failed to read label count");
    let n_labels = i32::from_be_bytes(buf4) as usize;

    let mut label_bytes = vec![0u8; n_labels];
    file.read_exact(&mut label_bytes).expect("Failed to read label data");

    label_bytes
        .iter()
        .map(|&label| {
            let mut one_hot = vec![0.0f64; 10];
            one_hot[label as usize] = 1.0;
            one_hot
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Returns the index of the maximum value in a slice (argmax).
fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .expect("argmax called on empty slice")
}

/// Computes accuracy over a subset of samples (indices into the full dataset).
///
/// Used to produce a cheap per-epoch training accuracy estimate on 1,000
/// randomly-chosen samples rather than the full 60,000.
fn accuracy_on_subset(
    network: &mut Network,
    images: &[Vec<f64>],
    labels: &[Vec<f64>],
    indices: &[usize],
) -> f64 {
    let mut correct = 0usize;
    for &idx in indices {
        let output = network.forward(images[idx].clone());
        if argmax(&output) == argmax(&labels[idx]) {
            correct += 1;
        }
    }
    correct as f64 / indices.len() as f64 * 100.0
}

// ---------------------------------------------------------------------------
// Inline training loop using CrossEntropyLoss
// ---------------------------------------------------------------------------

/// Trains `network` for one epoch using mini-batch SGD and CrossEntropyLoss.
///
/// Mirrors the logic in `src/train/trainer.rs` exactly, but substitutes
/// `CrossEntropyLoss` for `MseLoss` so that the Softmax output layer is
/// paired with the correct loss gradient.
///
/// Every `progress_every` batches a dot is printed to stdout and flushed
/// immediately, giving the user real-time feedback that training is running.
///
/// Returns the mean cross-entropy loss over all samples in the epoch.
fn train_epoch(
    network: &mut Network,
    inputs: &[Vec<f64>],
    expected_outputs: &[Vec<f64>],
    optimizer: &Sgd,
    batch_size: usize,
    progress_every: usize,
) -> f64 {
    let n = inputs.len();
    let mut total_loss = 0.0;

    // Shuffle sample indices so each epoch sees the data in a different order.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rand::thread_rng());

    let mut batch_count = 0usize;

    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let actual_batch_size = (batch_end - batch_start) as f64;

        // Zero-initialise accumulated gradient storage (one pair per layer).
        let mut acc_grads: Vec<(Matrix, Matrix)> = network.layers.iter()
            .map(|layer| (
                Matrix::zeros(layer.weights.rows, layer.weights.cols),
                Matrix::zeros(layer.biases.rows,  layer.biases.cols),
            ))
            .collect();

        // Accumulate gradients over all samples in this mini-batch.
        for &idx in &indices[batch_start..batch_end] {
            let input    = &inputs[idx];
            let expected = &expected_outputs[idx];

            // Forward pass — stores activations in each layer for backprop.
            let output = network.forward(input.clone());

            // Accumulate cross-entropy loss for reporting.
            total_loss += CrossEntropyLoss::loss(&output, expected);

            // Initial delta: combined Softmax + CE gradient = predicted - expected.
            // CrossEntropyLoss::derivative() returns exactly that vector.
            // The Softmax layer's derivative() returns 1.0, so the Hadamard
            // product inside compute_gradients() passes this delta through
            // unchanged — no double-application of the Jacobian.
            let error = CrossEntropyLoss::derivative(&output, expected);
            let mut delta = Matrix::from_data(vec![error]);

            // Backward pass — accumulate gradients layer by layer (reversed).
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
                    // Propagate delta to the previous layer through the weight
                    // matrix. `b_grad` here is the post-Hadamard layer_delta
                    // (shape 1×size_i), so multiplying by weights^T gives the
                    // error signal for layer i-1.
                    delta = b_grad.clone() * network.layers[i].weights.transpose();
                }

                acc_grads[i].0 = acc_grads[i].0.clone() + w_grad;
                acc_grads[i].1 = acc_grads[i].1.clone() + b_grad;
            }
        }

        // Average the accumulated gradients and apply the SGD update.
        let inv_batch = 1.0 / actual_batch_size;
        for (i, (w_acc, b_acc)) in acc_grads.into_iter().enumerate() {
            let w_avg = w_acc.map(|x| x * inv_batch);
            let b_avg = b_acc.map(|x| x * inv_batch);
            optimizer.step(&mut network.layers[i], w_avg, b_avg);
        }

        // Print a progress dot every `progress_every` batches and flush
        // immediately so the user sees it in real time (no buffering delay).
        batch_count += 1;
        if batch_count % progress_every == 0 {
            print!(".");
            io::stdout().flush().unwrap();
        }
    }

    total_loss / n as f64
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // --- Paths (relative to project root; cargo run --example runs from there) ---
    let train_images_path = "examples/mnist_data/train-images-idx3-ubyte";
    let train_labels_path = "examples/mnist_data/train-labels-idx1-ubyte";
    let test_images_path  = "examples/mnist_data/t10k-images-idx3-ubyte";
    let test_labels_path  = "examples/mnist_data/t10k-labels-idx1-ubyte";

    // --- Load data ---
    println!("Loading MNIST data...");
    let train_images = load_images(train_images_path);
    let train_labels = load_labels(train_labels_path);
    let test_images  = load_images(test_images_path);
    let test_labels  = load_labels(test_labels_path);

    println!(
        "  Training set: {} images, {} labels",
        train_images.len(),
        train_labels.len()
    );
    println!(
        "  Test set:     {} images, {} labels",
        test_images.len(),
        test_labels.len()
    );

    // --- Build network ---
    // 784 → 256 (ReLU) → 128 (ReLU) → 10 (Softmax)
    // Layer::new uses He init for ReLU layers and Xavier init for all others.
    let mut network = Network::new(vec![
        (256, 784, ActivationFunction::ReLU),
        (128, 256, ActivationFunction::ReLU),
        (10,  128, ActivationFunction::Softmax),
    ]);

    println!("\nNetwork architecture:");
    println!("  Input:   784 neurons (28x28 pixels, normalized)");
    println!("  Hidden1: 256 neurons — ReLU (He init)");
    println!("  Hidden2: 128 neurons — ReLU (He init)");
    println!("  Output:  10  neurons — Softmax (Xavier init)");
    println!("  Loss:    CrossEntropyLoss");
    println!("  Optimizer: SGD, lr = 0.01, batch_size = 32");

    // --- Training configuration ---
    let optimizer      = Sgd::new(0.01);
    let epochs         = 50;
    let batch_size     = 32;
    // Print a dot every 200 batches (≈ every 6,400 samples out of 60,000).
    // With 1,875 batches per epoch this gives ~9 dots per epoch so the user
    // can see forward motion within each epoch without flooding the terminal.
    let progress_every = 200;
    // Accuracy estimate uses 1,000 randomly-chosen training samples each
    // epoch — cheap enough to not noticeably slow training.
    let acc_subset_size = 1_000usize;

    // Pre-build the fixed subset of indices used for the per-epoch accuracy
    // estimate.  We reuse the same 1,000 indices every epoch for consistency.
    let mut rng = rand::thread_rng();
    let mut all_train_indices: Vec<usize> = (0..train_images.len()).collect();
    all_train_indices.shuffle(&mut rng);
    let acc_indices: Vec<usize> = all_train_indices[..acc_subset_size].to_vec();

    println!("\nTraining for {} epochs...", epochs);
    println!(
        "  Progress dots: one '.' per {} batches (~{} samples)",
        progress_every,
        progress_every * batch_size
    );
    println!(
        "  Training accuracy estimated on a fixed {}-sample subset each epoch.\n",
        acc_subset_size
    );

    // Pre-training baseline — expected ~10% for a random 10-class classifier.
    let baseline_acc = accuracy_on_subset(&mut network, &train_images, &train_labels, &acc_indices);
    println!("Pre-training accuracy (random weights): {:.2}%", baseline_acc);
    println!("  (Expected ~10% for a 10-class random classifier)\n");

    // Column header — widths match the format strings in the loop below.
    println!("{:>6}  {:>46}  {:>10}  {:>10}", "Epoch", "Progress", "CE Loss", "Train Acc");
    println!("{}", "─".repeat(80));

    for epoch in 1..=epochs {
        // Print epoch label and a leading bracket so the dots appear inline.
        print!("{:>6}  [", epoch);
        io::stdout().flush().unwrap();

        let loss = train_epoch(
            &mut network,
            &train_images,
            &train_labels,
            &optimizer,
            batch_size,
            progress_every,
        );

        // Close the dot-progress bracket, then append the scalar metrics.
        // The training accuracy is computed on the fixed 1,000-sample subset.
        let train_acc = accuracy_on_subset(
            &mut network,
            &train_images,
            &train_labels,
            &acc_indices,
        );

        // \r is NOT used here — we finish the line the dot-progress was on.
        println!("]  CE Loss: {:>10.6}  Train Acc: {:>6.2}%", loss, train_acc);
    }

    // --- Save model weights ---
    let model_dir = "examples/trained_models";
    let model_path = "examples/trained_models/mnist.json";
    std::fs::create_dir_all(model_dir).expect("Failed to create model directory");
    network.save_json(model_path).expect("Failed to save model");
    println!("\nModel saved to {}", model_path);

    // --- Evaluate on test set ---
    println!("\nEvaluating on test set ({} images)...", test_images.len());

    let mut correct = 0usize;
    let total = test_images.len();

    // Collect predictions for the first 10 images while we iterate.
    let mut sample_predictions: Vec<(usize, usize)> = Vec::new();

    for (i, (image, label)) in test_images.iter().zip(test_labels.iter()).enumerate() {
        let output    = network.forward(image.clone());
        let predicted = argmax(&output);
        let truth     = argmax(label);

        if predicted == truth {
            correct += 1;
        }

        if i < 10 {
            sample_predictions.push((truth, predicted));
        }
    }

    let accuracy = correct as f64 / total as f64 * 100.0;
    println!("  Correct: {}/{}", correct, total);
    println!("  Test accuracy: {:.2}%", accuracy);

    // --- Sample predictions ---
    println!("\nSample predictions (first 10 test images):");
    println!("{:>12}  {:>12}", "True Label", "Predicted");
    println!("{}", "-".repeat(27));
    for (truth, predicted) in &sample_predictions {
        println!("{:>12}  {:>12}", truth, predicted);
    }
}
