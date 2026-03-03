# ferrite-nn — Repo Curator Memory

## Crate Identity
- **Package name:** `ferrite-nn`
- **Lib name:** `ferrite_nn` — used in `use ferrite_nn::...` imports
- **Binary names:** `ferrite-nn` (src/main.rs, thin stub), `studio` (studio/main.rs, full browser IDE)
- **Working directory:** `/Users/radu/Developer/ferrite-nn`

## Module Structure (current and complete)
```
src/
  lib.rs              -- crate root; 27-line re-export surface (see Public API below)
  main.rs             -- thin binary; prints usage hint
  math/
    mod.rs
    matrix.rs         -- Matrix: zeros, he, xavier, random, transpose, map, from_data, +, -, *
  activation/
    mod.rs
    activation.rs     -- ActivationFunction: Sigmoid, ReLU, Identity, Softmax, Tanh,
                          LeakyReLU{alpha}, Elu{alpha}, Gelu, Swish
  layers/
    mod.rs
    dense.rs          -- Layer: new(), feed_from(), compute_gradients(), apply_gradients()
  network/
    mod.rs
    network.rs        -- Network: new(), forward(), save_json(), load_json(), from_spec()
    metadata.rs       -- ModelMetadata, InputType (Numeric/ImageGrayscale/ImageRgb)
    spec.rs           -- NetworkSpec, LayerSpec (serializable architecture descriptions)
  loss/
    mod.rs
    loss_type.rs      -- LossType enum: Mse, CrossEntropy, BinaryCrossEntropy, Mae, Huber
    mse.rs            -- MseLoss
    cross_entropy.rs  -- CrossEntropyLoss
    bce.rs            -- BceLoss
    mae.rs            -- MaeLoss
    huber.rs          -- HuberLoss (fixed delta=1.0)
  optim/
    mod.rs
    sgd.rs            -- Sgd: new(lr), step()
  train/
    mod.rs
    trainer.rs        -- train_network() — legacy MSE-only mini-batch SGD
    loop_fn.rs        -- train_loop() — multi-loss, SSE progress channel, stop flag
    epoch_stats.rs    -- EpochStats struct (serializable per-epoch metrics)
    train_config.rs   -- TrainConfig struct (epochs, batch_size, LossType, progress_tx, stop_flag)
studio/
  main.rs             -- tiny_http server on 127.0.0.1:7878; spawns one thread per request
  state.rs            -- StudioState + SharedState; imports Network, NetworkSpec, EpochStats
  render.rs           -- render_page() template engine; include_str! embeds studio.html
  routes.rs           -- dispatch() router + html_response/redirect/not_found helpers
  handlers/
    mod.rs
    architect.rs      -- GET/POST /architect; builds LayerSpec/NetworkSpec from form
    dataset.rs        -- GET /dataset; POST /dataset/upload, /upload-idx, /builtin
    train.rs          -- GET/POST /train/start, /train/stop; spawns train_loop thread
    train_sse.rs      -- GET /train/events (long-lived SSE stream)
    evaluate.rs       -- GET /evaluate, /evaluate/export; SVG loss curve, confusion matrix
    test.rs           -- GET/POST /test, /test/infer, /test/import-model
    models.rs         -- GET /models/{name}/download
  util/
    mod.rs
    form.rs           -- parse_form(), form_get(), url_decode()
    multipart.rs      -- multipart parser (no external dep): extract_file, extract_text_field, etc.
    csv.rs            -- parse_csv(); LabelMode enum; builtin_xor/circles/blobs datasets
    idx.rs            -- parse_idx_pair() — MNIST/IDX binary format parser
    sse.rs            -- SSE helpers (format_sse_event, write_sse, etc.) — currently dead_code
    image.rs          -- image_bytes_to_grayscale_input/rgb_input using `image` crate
  assets/
    studio.html       -- monolithic single-page template; 996 lines; {{TOKEN}} placeholders
examples/
  xor.rs              -- XOR gate demo; cargo run --example xor
  mnist.rs            -- MNIST 784→256→128→10 Softmax; cargo run --example mnist --release
examples/mnist_data/  -- IDX binary files (not committed; must be downloaded separately)
trained_models/       -- JSON model files; read by studio at runtime (CWD-relative path)
```

## Public API of `ferrite_nn` (lib.rs re-exports)
```
Matrix
ActivationFunction
Layer
Network
ModelMetadata, InputType
NetworkSpec, LayerSpec
MseLoss, CrossEntropyLoss, BceLoss, MaeLoss, HuberLoss
LossType
Sgd
train_network   (legacy)
train_loop      (preferred: multi-loss, SSE progress, stop flag)
EpochStats
TrainConfig
```
Note: `math::matrix::Matrix` is also accessible via `ferrite_nn::math::matrix::Matrix`
(examples/mnist.rs does this for `Matrix::zeros`).

## Implementation Status (all fully implemented — no stubs)
- Matrix ops, He/Xavier init: complete
- ActivationFunction (8 variants): complete
- Layer (forward + backprop): complete
- Network (new, forward, from_spec, save/load JSON): complete
- ModelMetadata / InputType / NetworkSpec / LayerSpec: complete
- All 5 loss functions + LossType: complete
- Sgd: complete
- train_network() (MSE-only legacy): complete
- train_loop() (multi-loss, validation, SSE, stop flag): complete
- EpochStats, TrainConfig: complete
- Studio (5 tabs, SSE training, CSV/IDX upload, image inference): complete

## Dependencies (Cargo.toml — single workspace)
- `rand 0.8.5` — weight init shuffling
- `serde 1` + `serde_json 1` — model/spec serialization, EpochStats SSE
- `tiny_http 0.12` — HTTP server for studio binary
- `image 0.24` (default-features=false, features=["png","jpeg","bmp","gif"]) — image inference

## Studio Architecture
- Single binary `studio` in same crate, path `studio/main.rs`
- `include_str!("assets/studio.html")` — template embedded at compile time
- Thread-per-request model; SSE handler takes ownership of TCP stream via `request.into_writer()`
- SharedState = `Arc<Mutex<StudioState>>` passed to every handler
- `trained_models/` read/written relative to CWD (must run from project root)

## Studio to Library Coupling (4 import sites)
- `state.rs`: `use ferrite_nn::{Network, NetworkSpec, EpochStats};`
- `handlers/architect.rs`: `use ferrite_nn::{ActivationFunction, LossType, NetworkSpec, LayerSpec};`
  + inline `ferrite_nn::ModelMetadata { ... }` usage
- `handlers/train.rs`: `use ferrite_nn::{Network, Sgd, LossType, TrainConfig, train_loop};`
  + `ferrite_nn::EpochStats` used as type annotation (3 more occurrences)
- `handlers/test.rs`: `use ferrite_nn::{ActivationFunction, InputType, Network};`
- `handlers/evaluate.rs`: `ferrite_nn::EpochStats` and `ferrite_nn::Network` as type annotations
  (no top-level use statement — uses fully-qualified paths)

## Key Conventions
- Weight init: He for ReLU, Xavier for all others (automatic in Layer::new)
- `train_loop()` is the preferred trainer; `train_network()` is legacy (MSE-only)
- Models saved/loaded via `Network::save_json(path)` / `Network::load_json(path)`
- `NetworkSpec::save_json()` / `load_json()` for architecture-only storage
- SSE progress: configure `TrainConfig::progress_tx` + optionally `stop_flag`
- Activation string ↔ enum mapping lives in `studio/handlers/architect.rs`

## Git State (as of Mar 1 2026)
- Branch: master (only branch; no remote configured)
- Modified (unstaged): studio/ files, studio/assets/studio.html, trained_models/
- Untracked: studio/util/idx.rs, trained_models/first_network__1_.json, trained_models/mnist.json
- .gitignore: `/target` only — trained_models/ and examples/mnist_data/ are NOT ignored

## Known Issues / Warnings
- `Layer.size` field may be unused (pre-existing compiler warning)
- `studio/util/sse.rs` has `#![allow(dead_code)]` — helpers defined but not yet used by train_sse.rs
- `train_network()` in trainer.rs is MSE-only; does not use `LossType`; kept for backward compat

## Roadmap (not yet implemented)
- Adam, RMSProp, momentum SGD
- Batch normalization
- Convolutional layers
- WASM inference
- Python bindings (PyO3)
- More datasets (FashionMNIST, CIFAR-10)
