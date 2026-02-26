# rust-mnist / ferrite-nn — Repo Curator Memory

## Crate Identity (updated after rename)
- **Package name:** `ferrite-nn` (was `rust-mnist`)
- **Lib name:** `ferrite_nn` (was `rust_mnist`) — used in `use ferrite_nn::...` imports
- **Binary name:** `ferrite-nn`
- **Working directory:** `/Users/radu/Developer/rust-mnist` (directory not renamed)

## Module Structure (current and complete)
```
src/
  lib.rs              -- crate root; declares all modules + convenience re-exports
  math/
    mod.rs            -- pub use matrix::Matrix
    matrix.rs         -- Matrix struct (Add, Sub, Mul, transpose, map, zeros, random, from_data)
  activation/
    mod.rs            -- pub use activation::ActivationFunction
    activation.rs     -- ActivationFunction enum (Sigmoid, ReLU, Identity) + function()/derivative()
  layers/
    mod.rs            -- pub use dense::Layer
    dense.rs          -- Layer struct; feed_from(), compute_gradients(), apply_gradients()
  network/
    mod.rs            -- pub use network::Network
    network.rs        -- Network struct; new() + forward()
  loss/
    mod.rs            -- pub use mse::MseLoss
    mse.rs            -- MseLoss; loss() + derivative()
  optim/
    mod.rs            -- pub use sgd::Sgd
    sgd.rs            -- Sgd; new() + step()
  train/
    mod.rs            -- pub use trainer::train_network
    trainer.rs        -- train_network() per-sample SGD loop
  main.rs             -- thin binary; prints usage hint
examples/
  xor.rs              -- XOR demo; `cargo run --example xor`
```

## Implementation Status (all fully implemented — no stubs)
- Matrix ops: complete
- ActivationFunction (Sigmoid, ReLU, Identity): complete
- Layer (forward + backprop): complete
- Network (forward pass): complete
- MseLoss (loss + derivative): complete
- Sgd (step): complete
- train_network(): complete

## Key Conventions
- Library logic in `src/`, demos in `examples/`
- Each module directory has a `mod.rs` that re-exports from sibling files
- `cargo run --example xor` to run the XOR demo
- Single external dependency: `rand = "0.8.5"` (weight initialization only)

## Roadmap (not yet implemented)
- Softmax activation + cross-entropy loss
- Adam optimizer
- Mini-batch training with gradient accumulation
- Xavier / He weight initialization
- MNIST data loader

## Known Warnings
- `Layer.size` field is never read (pre-existing; not a bug introduced by restructure)

## README
- README.md exists at repo root (created during pivot session)
- Covers: overview, features, quick start, architecture tree, roadmap, license (MIT)
