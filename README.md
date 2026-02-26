# ferrite-nn

A from-scratch neural network library in Rust — no external ML dependencies.

## Overview

`ferrite-nn` is a pure Rust implementation of a multi-layer perceptron (MLP) library built without any external machine learning dependencies. The goal is to be a clean, readable reference for how neural networks work under the hood: forward passes, backpropagation, gradient descent, and loss computation are all written explicitly from first principles.

The library is designed to be practical as well as educational. It is structured as a proper Rust library crate, with a working example (XOR) that demonstrates the full training loop converging to correct outputs.

## Features

- **Matrix operations** — 2D matrix struct with addition, subtraction, matrix multiplication, transpose, element-wise map, and constructors (`zeros`, `random`, `from_data`)
- **Activation functions** — `Sigmoid`, `ReLU`, and `Identity`, each with `function()` and `derivative()` methods; assigned per-layer at construction time
- **Dense (fully-connected) layer** — forward pass (`feed_from`), gradient computation (`compute_gradients`), and gradient application (`apply_gradients`); stores pre-activation values for correct backpropagation
- **Network** — builds an MLP from `(size, input_size, activation)` specs; `forward()` runs the full forward pass chaining all layers
- **MSE loss** — scalar mean-squared-error loss and per-output derivative
- **SGD optimizer** — vanilla stochastic gradient descent with a fixed learning rate
- **Training loop** — `train_network()` runs per-sample SGD over a batch of inputs and returns mean batch MSE loss

## Quick Start

Add `ferrite-nn` as a local path dependency in your `Cargo.toml`:

```toml
[dependencies]
ferrite-nn = { path = "../ferrite-nn" }
```

Then use it:

```rust
use ferrite_nn::{Network, Sgd, ActivationFunction, train_network};

fn main() {
    let mut network = Network::new(vec![
        (2, 2, ActivationFunction::Sigmoid),
        (1, 2, ActivationFunction::Sigmoid),
    ]);

    let inputs = vec![
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
        vec![0.0, 0.0],
    ];
    let expected_outputs = vec![
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
    ];

    let optimizer = Sgd::new(0.1);

    for epoch in 0..10000 {
        let loss = train_network(&mut network, &inputs, &expected_outputs, &optimizer);
        if epoch % 1000 == 0 {
            println!("Epoch {epoch}: loss = {loss:.6}");
        }
    }

    for input in &inputs {
        println!("Input: {:?} -> Output: {:.4}", input, network.forward(input.clone())[0]);
    }
}
```

To run the included XOR demo:

```sh
cargo run --example xor
```

After ~10 000 epochs the network converges to outputs near 0.95 / 0.05 for XOR.

## Architecture

```
src/
  lib.rs              -- crate root; declares all modules + convenience re-exports
  math/
    mod.rs
    matrix.rs         -- Matrix struct (Add, Sub, Mul, transpose, map, zeros, random, from_data)
  activation/
    mod.rs
    activation.rs     -- ActivationFunction enum (Sigmoid, ReLU, Identity)
  layers/
    mod.rs
    dense.rs          -- Layer struct; feed_from(), compute_gradients(), apply_gradients()
  network/
    mod.rs
    network.rs        -- Network struct; new() + forward()
  loss/
    mod.rs
    mse.rs            -- MseLoss; loss() + derivative()
  optim/
    mod.rs
    sgd.rs            -- Sgd; new() + step()
  train/
    mod.rs
    trainer.rs        -- train_network() full per-sample SGD training loop
  main.rs             -- thin binary entry point
examples/
  xor.rs              -- XOR classification demo; run with `cargo run --example xor`
```

The library uses a single external dependency (`rand 0.8`) for weight initialization only.

## Roadmap

Planned additions, in rough priority order:

- [ ] Softmax activation + cross-entropy loss (required for multi-class classification and MNIST)
- [ ] Adam optimizer
- [ ] Mini-batch training with gradient accumulation
- [ ] Xavier / He weight initialization strategies
- [ ] MNIST data loader

## License

MIT
