# ferrite-nn — Repo Curator Memory

## Crate Identity
- **Package name:** `ferrite-nn` (was `rust-mnist`)
- **Lib name:** `ferrite_nn` — used in `use ferrite_nn::...` imports
- **Binary name:** `ferrite-nn`
- **Working directory:** `/Users/radu/Developer/ferrite-nn` (old path was `/Users/radu/Developer/rust-mnist`)

## Module Structure (current and complete)
```
src/
  lib.rs              -- crate root; re-exports everything public
  math/
    mod.rs
    matrix.rs         -- Matrix: zeros, he, xavier, random, transpose, map, +, -, *
  activation/
    mod.rs
    activation.rs     -- ActivationFunction: Sigmoid, ReLU, Identity, Softmax
  layers/
    mod.rs
    dense.rs          -- Layer: new(), feed_from(), compute_gradients(), apply_gradients()
  network/
    mod.rs
    network.rs        -- Network: new(), forward(), save_json(), load_json()
  loss/
    mod.rs
    mse.rs            -- MseLoss: loss() + derivative()
    cross_entropy.rs  -- CrossEntropyLoss: numerically-stable CE paired with Softmax
  optim/
    mod.rs
    sgd.rs            -- Sgd: new(lr), step()
  train/
    mod.rs
    trainer.rs        -- train_network() mini-batch SGD loop
  main.rs             -- thin binary entry point
examples/
  xor.rs              -- XOR gate demo; `cargo run --example xor`
  mnist.rs            -- MNIST digit classifier; `cargo run --example mnist --release`
  gui.rs              -- local web inference server; `cargo run --example gui --release`
examples/mnist_data/  -- IDX binary files (downloaded separately, not committed)
examples/trained_models/ -- saved JSON model files (output of training runs)
```

## Implementation Status (all fully implemented — no stubs)
- Matrix ops (including He and Xavier constructors): complete
- ActivationFunction (Sigmoid, ReLU, Identity, Softmax): complete
- Layer (forward + backprop, auto weight init): complete
- Network (forward + save_json/load_json): complete
- MseLoss + CrossEntropyLoss: complete
- Sgd: complete
- train_network() with mini-batch support: complete

## Key Conventions
- Library logic in `src/`, demos in `examples/`
- Each module directory has a `mod.rs` that re-exports from sibling files
- Weight init is automatic: He for ReLU layers, Xavier for all others
- `train_network()` signature: `(network, inputs, expected, optimizer, epoch, batch_size)`
- Models saved/loaded via `Network::save_json(path)` / `Network::load_json(path)`

## Dependencies
- `rand 0.8.5` — weight initialization
- `serde 1` + `serde_json 1` — model serialization
- `tiny_http 0.12` — web GUI server (dev-dependency only)

## Examples
- `xor`: 2→4(Sigmoid)→1(Sigmoid), MSE, SGD lr=0.1, 10k epochs
- `mnist`: 784→256(ReLU)→128(ReLU)→10(Softmax), CrossEntropyLoss, SGD lr=0.01, batch=32, 50 epochs; ~97% test accuracy; saves to `examples/trained_models/mnist.json`
- `gui`: tiny_http server on 127.0.0.1:7878; loads any model from `examples/trained_models/`; auto-formats output by activation type

## Custom Subagents
`.claude/agents/` contains:
- `ferrite-nn-maintainer` — library engineering, primitives, bug fixes
- `ferrite-nn-trainer` — training runs, data curation, results
- `ferrite-nn-gui` — web GUI server and HTML template

## README
- README.md at repo root; full rewrite in Feb 2026
- Tone: friendly, a few emojis, technically accurate, no over-selling
- Sections: tagline, examples table, getting started, architecture, web GUI, training your own model, future plans, license
- Does NOT use a "Features" bullet list — uses a table for examples and code snippets for API

## Roadmap (not yet implemented)
- Adam, RMSProp, momentum SGD
- Batch normalization
- Convolutional layers
- WASM inference
- Python bindings (PyO3)
- More datasets (FashionMNIST, CIFAR-10)

## Known Warnings
- `Layer.size` field may be unused (pre-existing; not a bug)
