# Ferrite-NN Maintainer Memory

## Module Structure

```
src/
  lib.rs                   — top-level crate root; re-exports all public types
  main.rs                  — binary entry point (unused for library use)
  math/
    mod.rs                 — declares math::matrix, re-exports Matrix
    matrix.rs              — Matrix struct (Vec<Vec<f64>>) with ops, constructors
  activation/
    mod.rs                 — declares activation::activation, re-exports ActivationFunction
    activation.rs          — ActivationFunction enum (Sigmoid, ReLU, Identity, Softmax)
  layers/
    mod.rs                 — declares layers::dense, re-exports Layer
    dense.rs               — Layer struct: forward (feed_from), backward (compute_gradients), apply_gradients
  network/
    mod.rs                 — declares network::network, re-exports Network
    network.rs             — Network: Vec<Layer>, forward()
  loss/
    mod.rs                 — declares mse + cross_entropy, re-exports MseLoss, CrossEntropyLoss
    mse.rs                 — MseLoss: loss(), derivative()
    cross_entropy.rs       — CrossEntropyLoss: loss(), derivative() (Softmax+CE combined gradient)
  optim/
    mod.rs                 — declares optim::sgd, re-exports Sgd
    sgd.rs                 — Sgd: step() calls layer.apply_gradients()
  train/
    mod.rs                 — declares train::trainer, re-exports train_network
    trainer.rs             — train_network(network, inputs, expected, optimizer, batch_size) -> f64
examples/
  xor.rs                   — XOR demo; use batch_size=1 for online SGD
```

## Key Patterns & Conventions

- Module pattern: `mod.rs` declares sub-module and re-exports; implementation lives in a `.rs` file with same name.
- All public types are also re-exported from `src/lib.rs` for convenience.
- Matrix shape docs use `(rows, cols)`; `data` is `Vec<Vec<f64>>`, row-major.
- `Layer` stores `pre_neurons` (pre-activation z) for correct derivative in backprop.
- `compute_gradients()` returns `(weights_grad, biases_grad)` — caller accumulates.
- `apply_gradients()` is called by `Sgd::step()` with averaged grad and lr scaling.
- `train_network()` signature: `(network, inputs, expected_outputs, optimizer, batch_size)`.

## Activation: Softmax Special Cases

- `Softmax` is NOT element-wise; `Layer::feed_from()` has a special match arm for it.
- Numerically stable softmax: subtract `max(z)` before `exp`.
- `ActivationFunction::Softmax.derivative()` returns `1.0` — combined CE gradient already encodes `predicted - expected`; returning 1.0 prevents double-application.
- `function()` for Softmax panics with an informative message (should never be called directly).

## Weight Initialization

- `Matrix::random()` — uniform [-1,1], legacy only.
- `Matrix::he(rows, cols)` — N(0, sqrt(2/cols)), use before ReLU.
- `Matrix::xavier(rows, cols)` — N(0, sqrt(1/cols)), use before Sigmoid/Tanh/Identity/Softmax.
- Normal sampling uses Box-Muller (no rand_distr dependency).
- `Layer::new()` auto-selects: ReLU → He, everything else → Xavier. Biases init to zero.

## Loss Functions

- `MseLoss`: used in trainer by default; derivative = `predicted - expected`.
- `CrossEntropyLoss`: for Softmax output layers; derivative = `predicted - expected` (same form, different semantics — the Softmax Jacobian is already folded in).
- Epsilon guard in CE loss: `eps = 1e-12` inside `ln()`.

## Dependencies

- `rand = "0.8.5"` — only external dependency. No rand_distr.
- Box-Muller is implemented inline in `matrix.rs` for normal sampling.

## Testing

- No dedicated test files yet; correctness verified via `cargo run --example xor`.
- XOR should reach loss < 0.01 within ~6000 epochs with lr=0.1, batch_size=1.
- See `patterns.md` for finite-difference gradient check recipe (not yet implemented).
