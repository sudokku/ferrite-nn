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
studio/
  main.rs                  — HTTP server entry; SharedState; request dispatch loop
  state.rs                 — StudioState, TrainingStatus, DatasetState, Hyperparams, FlashMessage
  render.rs                — render_page(Page, tab_unlock, training_running, fill_closure) -> String
  routes.rs                — dispatch() takes &mut Request; match on (method, path)
  handlers/
    mod.rs                 — declares all handler modules
    architect.rs           — GET/POST /architect, /architect/save
    dataset.rs             — GET/POST /dataset, /dataset/upload, /dataset/builtin
    train.rs               — GET/POST /train, /train/start, /train/stop
    train_sse.rs           — GET /train/events (SSE; takes Request by value for streaming)
    evaluate.rs            — GET /evaluate, /evaluate/export
    test.rs                — GET/POST /test, /test/infer
    models.rs              — GET /models/{name}/download
  util/
    mod.rs                 — declares all util modules
    form.rs                — url_decode, parse_form, form_get
    multipart.rs           — extract_boundary, multipart_extract_file, extract_text_field, extract_all_text_fields
    csv.rs                 — parse_csv (LabelMode: ClassIndex/OneHot), builtin_xor/circles/blobs
    sse.rs                 — SSE helpers (mostly unused; train_sse writes raw HTTP)
    image.rs               — image_bytes_to_grayscale_input, image_bytes_to_rgb_input
  assets/
    studio.html            — Single-page 5-tab studio (embedded via include_str!)
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

- `SharedState = Arc<Mutex<StudioState>>` — locked only at the start/end of handlers, never during I/O.
- SSE handler clones the `Arc<Mutex<Receiver<EpochStats>>>` out before its receive loop.
- Tab unlock bitmask: bit 0=Architect (always), 1=Dataset (spec saved), 2=Train (dataset loaded), 3=Evaluate (done/stopped), 4=Test (always).
- POST-Redirect-GET pattern for all form submissions.
- Trained models saved to `trained_models/<name>.json` (project root).
- `studio.html` uses `{{PLACEHOLDER}}` tokens; `render.rs` blanks any unfilled ones.

## TrainingStatus enum (state.rs)

- `Stopped` variant **removed**. Stopping training now produces `Done { was_stopped: true, .. }`.
- `Done` fields: `model_path: String`, `elapsed_total_ms: u64`, `was_stopped: bool`.
- Model is always saved after training loop, regardless of whether the user clicked Stop.
- `tab_unlock_mask()` only matches `Done { .. }` (Stopped was removed).
- All handlers that previously matched `Stopped` must now check `Done { was_stopped, .. }`.
- SSE: emits `event: stopped` (with `model_path`) when `was_stopped=true`, `event: done` otherwise.
- XOR built-in dataset forces `val_split=0` (4 samples — validation split is misleading).
- Test tab has `POST /test/import-model` route; uses `find_subsequence`/`split_on` from multipart util.
- Train done card: `build_done_stats` adds `id="done-stats-js"` and "Saved to:" path line.
- studio.html: `restoreTrainDone()` repopulates done card from `sessionStorage` on client tab switch.

## Dependencies

- `rand = "0.8.5"`, `serde/serde_json = "1"`, `tiny_http = "0.12"`, `image = "0.24"` (all in `[dependencies]`).
- No `[dev-dependencies]` section.
- `tiny_http::Request::into_writer()` returns `Box<dyn Write + Send>` directly (not Result).

## Known Patterns / Gotchas

- Hex color codes in `format!` strings (e.g. `"#1e40af"`) cause Rust 2021 "unknown prefix" errors. Build SVG via string concatenation with color constants instead.
- `Network` and `Layer` now derive `Clone` (added to support `evaluate.rs` confusion matrix).
- The SSE handler writes raw HTTP headers manually (tiny_http `into_writer` bypasses response builder).
- When draining `mpsc::Receiver` from `Arc<Mutex<Receiver>>` in a struct, collect to a local Vec first to avoid borrow conflicts with other fields.
