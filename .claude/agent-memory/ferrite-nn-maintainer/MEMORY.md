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
    activation.rs          — ActivationFunction enum (Sigmoid, ReLU, Identity, Softmax) + PartialEq
  layers/
    mod.rs                 — declares layers::dense, re-exports Layer
    dense.rs               — Layer struct (Clone, Debug, Serialize, Deserialize): feed_from, compute_gradients, apply_gradients
  network/
    mod.rs                 — declares network, metadata, spec; re-exports Network, NetworkSpec, LayerSpec
    network.rs             — Network (Clone): Vec<Layer>, forward(), save_json(), load_json(), from_spec()
    metadata.rs            — ModelMetadata, InputType
    spec.rs                — NetworkSpec, LayerSpec (Serialize, Deserialize): save_json(), load_json()
  loss/
    mod.rs                 — declares mse + cross_entropy + loss_type; re-exports all
    mse.rs                 — MseLoss: loss(), derivative()
    cross_entropy.rs       — CrossEntropyLoss: loss(), derivative()
    loss_type.rs           — LossType enum (Mse, CrossEntropy); Serialize/PartialEq
  optim/
    mod.rs                 — declares optim::sgd, re-exports Sgd
    sgd.rs                 — Sgd: step() calls layer.apply_gradients()
  train/
    mod.rs                 — declares trainer, epoch_stats, train_config, loop_fn; re-exports all
    trainer.rs             — train_network() (MSE only, legacy)
    epoch_stats.rs         — EpochStats struct (Serialize/Deserialize)
    train_config.rs        — TrainConfig: epochs, batch_size, loss_type, progress_tx, stop_flag
    loop_fn.rs             — train_loop(): full training loop with SSE channel + stop flag
crates/ferrite-studio/src/
  main.rs                  — tokio::main; builds axum Router; binds 127.0.0.1:7878
  state.rs                 — StudioState, TrainingStatus, DatasetState, Hyperparams, FlashMessage
  routes.rs                — SharedState type alias; build_router() with all /api/ routes
  handlers/
    mod.rs                 — declares all handler modules
    architect.rs           — GET/POST /api/architect, /api/architect/save (JSON)
    dataset.rs             — GET/POST /api/dataset, /api/dataset/upload, /api/dataset/upload-idx, /api/dataset/builtin
    train.rs               — GET/POST /api/train, /api/train/start, /api/train/stop
    train_sse.rs           — GET /api/train/events (axum Sse<BoxStream<...>>)
    evaluate.rs            — GET /api/evaluate, /api/evaluate/export
    test.rs                — GET/POST /api/test, /api/test/infer, /api/test/import-model
    models.rs              — GET /api/models, /api/models/:name/download
  util/
    mod.rs                 — declares: csv, idx, image (form/multipart/sse deleted)
    csv.rs                 — parse_csv (LabelMode: ClassIndex/OneHot), builtin_xor/circles/blobs
    idx.rs                 — parse_idx_pair(&image_bytes, &label_bytes, n_classes) -> Result<...>
    image.rs               — image_bytes_to_grayscale_input, image_bytes_to_rgb_input
  assets/
    studio.html            — Dead weight (SSR removed; React SPA replaces it in Phase 2)
examples/
  xor.rs                   — XOR demo; use batch_size=1 for online SGD
  mnist.rs                 — MNIST classifier; saves to trained_models/mnist.json
trained_models/            — Project-root model storage (NOT examples/trained_models/)
```

## Key Patterns & Conventions

- Module pattern: `mod.rs` declares sub-module and re-exports; implementation in `.rs` file.
- All public types re-exported from `src/lib.rs` for convenience.
- Matrix shape docs use `(rows, cols)`; `data` is `Vec<Vec<f64>>`, row-major.
- `Layer` stores `pre_neurons` (pre-activation z) for correct derivative in backprop.
- `compute_gradients()` returns `(weights_grad, biases_grad)` — caller accumulates.
- `apply_gradients()` is called by `Sgd::step()` with averaged grad and lr scaling.
- `train_network()` signature: `(network, inputs, expected_outputs, optimizer, batch_size)`.
- **POST handlers take `&mut Request`** so routes.rs retains ownership for `request.respond()`.
- **SSE handler takes `Request` by value** (calls `into_writer()` for raw TCP streaming).
- `train_loop()` dispatches on `LossType`; supports `progress_tx` + `stop_flag`.

## Activation: Softmax Special Cases

- `Softmax` is NOT element-wise; `Layer::feed_from()` has a special match arm for it.
- Numerically stable softmax: subtract `max(z)` before `exp`.
- `ActivationFunction::Softmax.derivative()` returns `1.0` — combined CE gradient already encodes `predicted - expected`.
- `function()` for Softmax panics (never call directly).
- `ActivationFunction` now derives `PartialEq` (added for studio validation checks).

## Weight Initialization

- `Matrix::he(rows, cols)` — N(0, sqrt(2/cols)), use before ReLU.
- `Matrix::xavier(rows, cols)` — N(0, sqrt(1/cols)), use before Sigmoid/Tanh/Identity/Softmax.
- `Layer::new()` auto-selects: ReLU → He, everything else → Xavier. Biases init to zero.
- `Network::from_spec(spec)` builds a freshly-initialized network from a `NetworkSpec`.

## Loss Functions

- `MseLoss`: used in trainer by default; derivative = `predicted - expected`.
- `CrossEntropyLoss`: for Softmax output layers; derivative = `predicted - expected`.
- Epsilon guard in CE loss: `eps = 1e-12` inside `ln()`.
- `LossType` enum: `Mse` / `CrossEntropy`; studio enforces Softmax ↔ CrossEntropy consistency.

## Studio Architecture

- `SharedState = Arc<Mutex<StudioState>>` defined in `routes.rs` — locked only at the start/end of handlers, never during I/O or `.await`.
- All endpoints under `/api/` prefix, return JSON (no HTML).
- Tab unlock bitmask: bit 0=Architect (always), 1=Dataset (spec saved), 2=Train (dataset loaded), 3=Evaluate (done), 4=Test (always).
- No POST-Redirect-GET: handlers return JSON directly.
- Trained models saved to `trained_models/<name>.json` (project root, relative to CWD).
- `render.rs` deleted; `studio.html` is dead weight (React SPA replaces it in Phase 2).

## TrainingStatus enum (state.rs)

- Variants: `Idle`, `Running { stop_flag, epoch_rx, total_epochs }`, `Done { model_path, elapsed_total_ms, was_stopped }`, `Failed { reason }`.
- `Running.epoch_rx` is `Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<EpochStats>>>` (NOT std).
- Stopping training produces `Done { was_stopped: true, .. }` — model is always saved.
- SSE: emits `event: stopped` (with `model_path`) when `was_stopped=true`, `event: done` otherwise.
- XOR built-in dataset forces `val_split=0` (4 samples — validation split is misleading).

## Dependencies

### ferrite-nn (library crate)
- `rand = "0.8.5"`, `serde/serde_json = "1"`, `image = "0.24"` (all in `[dependencies]`).
- No `[dev-dependencies]` section.

### ferrite-studio (binary crate)
- `axum = "0.7"` (NOT 0.8 — axum 0.8 requires rustc 1.78, axum 0.7 MSRV is 1.66).
- `tower-http = "0.5"` (MSRV 1.66, matching axum 0.7).
- `tokio = "1"` with `features = ["full"]`.
- `futures = "0.3"` for stream combinators in SSE handler.
- `image = "0.24"` with `default-features = false, features = ["png","jpeg","bmp","gif"]`.
- axum 0.7 uses `:name` path param syntax; axum 0.8 uses `{name}` — use `:name` for compatibility.

## Dataset: IDX Format Support

- `crates/ferrite-studio/src/util/idx.rs`: `parse_idx_pair` validates IDX3 image + IDX1 label files and returns `(inputs, labels)` compatible with `build_dataset_state`.
- IDX files uploaded via `POST /api/dataset/upload-idx` multipart with fields `images_file` and `labels_file` — handled by axum `Multipart` extractor.
- `MAX_IDX_BYTES = 100 MB` (MNIST train ~47 MB; allow headroom).
- IDX handler derives `source_name` as `"IDX upload (N samples, SxS px, C classes)"` using `sqrt(n_pixels)` for a best-effort square dimension.

## Studio Architecture (Phase 1: axum REST API)

- `ferrite-studio` is now a JSON REST API served by axum 0.7 on port 7878.
- All routes prefixed with `/api/` — no HTML is served.
- `SharedState = Arc<Mutex<StudioState>>` lives in `routes.rs`.
- axum `State<SharedState>` extractor used in all handlers.
- `TrainingStatus::Running` uses `tokio::sync::mpsc::Receiver` (NOT `std::sync::mpsc`) so the async SSE handler can recv without blocking.
- The training background thread uses `std::sync::mpsc::Sender` → relay thread bridges to `tokio::sync::mpsc::Sender` via `blocking_send`.
- SSE handler (`train_sse.rs`) uses `axum::response::sse::{Sse, Event, KeepAlive}` + `futures::stream::{StreamExt, unfold, iter}`.
- The two SSE match arms (Running vs. not-Running) use `Box<Pin<...>>` (`SseStream` alias) to unify the concrete stream types.
- `axum::extract::Multipart` consumes the body; must be the last extractor. When `State<S>` + `Multipart` is used together, the `State` arg must NOT be shadowed or locked in the same expression — keep lock scopes inside the body, dropped before any `.await`.
- Deleted files: `render.rs`, `util/form.rs`, `util/multipart.rs`, `util/sse.rs`.
- `util/mod.rs` now only declares: `csv`, `idx`, `image`.
- `handlers/models.rs` `handle_list` takes NO `State` — reads filesystem directly.
- Dataset handlers (`handle_upload`, `handle_upload_idx`) return the full `DatasetResponse` JSON on success, not a redirect.
- `handle_infer` uses `State(_state)` (underscore) because it doesn't currently need state; keep the param for future extension.

## Known Patterns / Gotchas

- Hex color codes in `format!` strings (e.g. `"#1e40af"`) cause Rust 2021 "unknown prefix" errors. Build SVG via string concatenation with color constants instead.
- `Network` and `Layer` now derive `Clone` (added to support `evaluate.rs` confusion matrix).
- axum 0.7 `Handler<T, S>` blanket impls require the state type `S: Clone`; `Arc<Mutex<T>>` satisfies this.
- When both `State<S>` and `Multipart` are handler extractors and the handler also locks the state, avoid holding the `MutexGuard` across any `.await` point — the compiler may reject the future as non-`Send`.
- `tokio::sync::Mutex` (not `std`) is used for the epoch receiver in `TrainingStatus::Running` to allow locking inside an async context.
- The SSE stream type must be boxed (`Pin<Box<dyn Stream<...>>>`) when different match arms produce structurally different stream types — otherwise the function won't compile due to return type mismatch.
