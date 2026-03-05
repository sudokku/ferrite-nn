# ferrite-nn

A neural network library built from scratch in Rust — no PyTorch, no TensorFlow, no external ML
dependencies. Just matrices, dot products, and backpropagation, all the way down.

`ferrite-nn` is structured as a Cargo workspace with two crates: the core ML library and a local
studio server that exposes the library as a JSON REST API for browser-based tooling.

---

## Workspace layout

```
ferrite-nn/                         workspace root
  Cargo.toml                        workspace manifest
  crates/
    ferrite-nn/                     the neural network library (pure Rust)
      src/
        lib.rs
        math/matrix.rs              Matrix ops (zeros, he, xavier, transpose, map, +, -, *)
        activation/activation.rs    ActivationFunction enum (9 variants)
        layers/dense.rs             Layer: forward pass + backprop
        network/
          network.rs                Network: build, train, forward, save/load JSON
          metadata.rs               ModelMetadata, InputType
          spec.rs                   NetworkSpec, LayerSpec
        loss/                       MseLoss, CrossEntropyLoss, BceLoss, MaeLoss, HuberLoss
        optim/sgd.rs                Sgd optimizer
        train/
          trainer.rs                train_network() — legacy mini-batch SGD
          loop_fn.rs                train_loop() — multi-loss, SSE channel, stop flag
          epoch_stats.rs            EpochStats struct
          train_config.rs           TrainConfig struct
      examples/
        xor.rs                      XOR gate demo
        mnist.rs                    MNIST digit classifier (~97% accuracy)
    ferrite-studio/                 JSON REST API server (axum + tokio)
      src/
        main.rs                     tokio::main, listens on 127.0.0.1:7878
        routes.rs                   all /api/* routes
        state.rs                    StudioState (Arc<Mutex<...>>)
        handlers/
          architect.rs              GET/POST /api/architect
          dataset.rs                GET/POST /api/dataset/*
          train.rs                  POST /api/train/start|stop
          train_sse.rs              GET /api/train/events (SSE stream)
          evaluate.rs               GET /api/evaluate
          test.rs                   POST /api/test/infer
          models.rs                 GET /api/models, model download
        util/
          idx.rs                    IDX binary format parser (MNIST data files)
          csv.rs                    CSV dataset parser
          image.rs                  image preprocessing for inference
  docs/
    api-reference.md                full REST API reference
    ferrite-studio-CLAUDE.md        CLAUDE.md template for the frontend repo
  ROADMAP.md
```

Weight initialization is automatic: He for ReLU layers, Xavier for all others.

---

## Quick start: library

Require Rust 1.75 or newer.

```sh
git clone https://github.com/sudokku/ferrite-nn
cd ferrite-nn
```

Add `ferrite-nn` as a path dependency in your own crate:

```toml
[dependencies]
ferrite-nn = { path = "../ferrite-nn/crates/ferrite-nn" }
```

Build and run inference in a few lines:

```rust
use ferrite_nn::{Network, ActivationFunction, Sgd, train_loop, TrainConfig, LossType};
use std::sync::{Arc, atomic::AtomicBool};

fn main() {
    // 784-input -> 256 ReLU -> 128 ReLU -> 10 Softmax
    let mut network = Network::new(vec![
        (256, 784, ActivationFunction::ReLU),
        (128, 256, ActivationFunction::ReLU),
        (10,  128, ActivationFunction::Softmax),
    ]);

    let config = TrainConfig {
        epochs: 50,
        batch_size: 32,
        learning_rate: 0.01,
        loss_type: LossType::CrossEntropy,
        progress_tx: None,
        stop_flag: Some(Arc::new(AtomicBool::new(false))),
    };

    train_loop(&mut network, &inputs, &labels, &config);

    network.save_json("trained_models/my_model.json").unwrap();
}
```

Reload and run inference at any time:

```rust
let mut net = Network::load_json("trained_models/my_model.json").unwrap();
let output = net.forward(input);
```

---

## Quick start: studio server

The `ferrite-studio` crate wraps the library as a JSON REST API server. Run it from the workspace
root:

```sh
cargo run --bin studio --release
```

The server starts on `http://127.0.0.1:7878` and exposes all the studio endpoints documented in
`docs/api-reference.md`. It handles training jobs, SSE progress streaming, model management,
dataset upload, inference, and evaluation — all as JSON.

---

## Studio frontends

The studio server is designed to work with any HTTP client or web frontend. There are three ways
to interact with it:

**Official frontend (recommended for full use)**
The `ferrite-studio` repository is a separate project providing a React SPA (Vite + TypeScript +
shadcn/ui + Recharts) and a Python FastAPI layer for user accounts, authentication (JWT + OAuth),
and model sharing. It talks to the Rust server over the REST API documented in
`docs/api-reference.md`.

**Custom frontend**
Build your own UI against the REST API. The full endpoint reference is in
[`docs/api-reference.md`](docs/api-reference.md).

**Minimal inference GUI**
`examples/gui.rs` is a planned self-contained inference server using `tiny_http` — no studio
server required. It serves a simple browser form for running predictions against saved models.
(Not yet implemented; tracked in ROADMAP.md.)

---

## Features

**Activation functions**
Sigmoid, ReLU, Tanh, LeakyReLU (configurable alpha), ELU (configurable alpha), GELU, Swish,
Softmax, Identity

**Loss functions**
Mean Squared Error, Cross-Entropy, Binary Cross-Entropy, Mean Absolute Error, Huber

**Model serialization**
Networks save and load as JSON. `ModelMetadata` (including `InputType`) is embedded in the file
for self-describing models — useful when an inference server needs to know whether to expect
numeric inputs, grayscale images, or RGB images.

**IDX binary format**
Built-in parser for MNIST-style IDX files — loads training images and labels directly.

**Image preprocessing**
The studio server handles image uploads for inference: `load_from_memory` -> resize (Lanczos3)
-> normalize to [0, 1] before forwarding to the network.

**Training loop**
`train_loop()` supports any `LossType`, accepts an optional SSE progress channel, and an optional
atomic stop flag for cancellation from the studio.

---

## Examples

### XOR demo

No setup needed:

```sh
cargo run --example xor
```

A small network converges on XOR in ~10 000 epochs. Output values settle near 0.95 and 0.05.

### MNIST digit classifier

Download the four IDX binary files from
[Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/) and place them in
`examples/mnist_data/`:

```
examples/mnist_data/
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
```

Train:

```sh
cargo run --example mnist --release
```

Training prints per-epoch loss and accuracy. After 50 epochs the model is saved to
`trained_models/mnist.json` and achieves approximately 97% test accuracy.

---

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the full plan, including:

- Phase 1: migrate studio to Axum REST API (in progress)
- Phase 2: React SPA frontend (replaces the current HTML monolith)
- Phase 3: Python FastAPI auth layer, user accounts, model sharing (`ferrite-studio` repo)
- Phase 4: Docker Compose deployment (nginx + FastAPI + Rust service + PostgreSQL)

---

## License

MIT
