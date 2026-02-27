# ferrite-nn Agent Memory

## Project Overview
Ferrite-nn is a from-scratch neural network library in Rust. Located at `/Users/radu/Developer/ferrite-nn`.

## Key File Paths
- Public API: `src/lib.rs`
- Matrix ops: `src/math/matrix.rs`
- Dense layer: `src/layers/dense.rs`
- Network struct: `src/network/network.rs`
- Trainer: `src/train/trainer.rs`
- MSE loss: `src/loss/mse.rs`
- CrossEntropy loss: `src/loss/cross_entropy.rs`
- SGD optimizer: `src/optim/sgd.rs`
- Activations: `src/activation/activation.rs`
- Examples: `examples/xor.rs`, `examples/mnist.rs`

## Architecture Summary (verified Feb 2026)
- Matrix: Vec<Vec<f64>> backed, row-major. Ops: zeros, random, he, xavier, transpose, map, from_data, Add/Sub/Mul.
- Layer: Dense only. Forward: z = xW + b, a = σ(z). Stores pre_neurons (z) for backprop. Weights shape [input_size, size].
  - ReLU layers use He init; all other activations use Xavier init. Biases always zero-initialized.
  - Softmax is applied vector-wise in feed_from(), not element-wise. Softmax.derivative() returns 1.0 (identity pass-through for combined CE gradient).
- Network: Sequential Vec<Layer>. forward() threads input through all layers. layers field is pub.
- Trainer (train_network): Mini-batch SGD with shuffling. HARDCODES MseLoss. Signature: fn train_network(network: &mut Network, inputs: &[Vec<f64>], expected_outputs: &[Vec<f64>], optimizer: &Sgd, batch_size: usize) -> f64
- Loss: MseLoss AND CrossEntropyLoss both exist. CE gradient = predicted - expected (combined Softmax+CE shortcut).
- Optimizer: Sgd::new(lr: f64). step(&mut Layer, w_grad: Matrix, b_grad: Matrix).
- Activations: Sigmoid, ReLU, Identity, Softmax. No Tanh.

## Trainer Hardcodes MseLoss — Key Pattern
When a task requires CrossEntropyLoss (e.g. Softmax output), do NOT call train_network().
Instead, inline the training loop in the example (Option B), substituting CrossEntropyLoss::loss() and CrossEntropyLoss::derivative() for MseLoss counterparts. The rest of the backward pass logic is identical. See examples/mnist.rs for the full pattern.

## Backward Pass Details
The b_grad returned by layer.compute_gradients() is the post-Hadamard layer_delta (shape 1×size).
To propagate delta to layer i-1: delta = b_grad * network.layers[i].weights.transpose().
For Softmax+CE: initial delta = CrossEntropyLoss::derivative(&output, &expected) wrapped in Matrix::from_data(vec![error]). Softmax.derivative()=1.0 makes Hadamard a no-op, so the combined gradient passes through correctly.

## Public Re-exports in lib.rs
use ferrite_nn::{Matrix, ActivationFunction, Layer, Network, MseLoss, CrossEntropyLoss, Sgd, train_network};
Also accessible: ferrite_nn::math::matrix::Matrix (needed for inlined training loops).

## Cargo.toml Pattern
Each example needs a [[example]] section with name and path. Both xor and mnist are registered.

## stdout Flushing Pattern
`println!` flushes on newline. `print!` (no newline) does NOT flush automatically.
Always follow `print!` with `io::stdout().flush().unwrap()` for real-time output.
Import: `use std::io::{self, Read, Write};` — Read and Write are separate traits and coexist.

## Per-Epoch Progress Pattern (mnist.rs)
- `train_epoch` accepts a `progress_every: usize` parameter.
- Prints "." every N batches inside the loop + flush immediately.
- Main loop prints `"Epoch  ["` before calling train_epoch, then `"]  loss  acc%"` after.
- Per-epoch accuracy: run forward-only on a fixed 1,000-sample index subset (shuffled once before training). This avoids the cost of evaluating all 60,000 training samples each epoch.

## What Works Well
- Small binary classification (XOR, AND, OR) via train_network() + MseLoss
- MNIST-scale multi-class classification via inlined loop + CrossEntropyLoss + Softmax

## Detailed Notes
- See `debugging.md` for earlier architecture analysis details.
