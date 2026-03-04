# ferrite-nn Studio — REST API Reference

> **Base URL (local dev):** `http://127.0.0.1:7878`
> **Server:** Rust / axum 0.7, `cargo run --bin studio --release`
> **Auth:** None yet (Phase 3 adds FastAPI JWT gateway in front of this service)
> **Content-Type:** All request bodies are `application/json` unless noted as `multipart/form-data`
> **All responses are JSON** unless noted as file downloads or SSE streams

---

## Global response conventions

Every endpoint returns JSON. Errors have the shape:
```json
{ "error": "human-readable message" }
```

Successful mutations return at minimum:
```json
{ "ok": true }
```

The `tab_unlock` field present on several GET responses is a bitmask (u8) used by the frontend to know which workflow tabs are currently accessible:
- bit 0 (0x01) — Architect (always unlocked)
- bit 1 (0x02) — Dataset (unlocked when a spec is saved)
- bit 2 (0x04) — Train (unlocked when a dataset is loaded)
- bit 3 (0x08) — Evaluate (unlocked when training is Done or Stopped)
- bit 4 (0x10) — Test (always unlocked)

---

## Architect

### `GET /api/architect`

Returns the current network architecture spec, hyperparameters, and any pending flash message.

**Response:**
```json
{
  "spec": {
    "name": "my_model",
    "description": "MNIST classifier",
    "input_size": 784,
    "loss_type": "cross_entropy",
    "layers": [
      { "neurons": 256, "activation": "relu" },
      { "neurons": 128, "activation": "relu" },
      { "neurons": 10,  "activation": "softmax" }
    ],
    "metadata": {
      "input_type": { "type": "ImageGrayscale", "width": 28, "height": 28 },
      "output_labels": ["0","1","2","3","4","5","6","7","8","9"],
      "description": "..."
    }
  },
  "hyperparams": {
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 50
  },
  "tab_unlock": 3,
  "flash": null
}
```

`spec` and `hyperparams` are `null` when no architecture has been saved yet.
`flash` is `{ "kind": "success" | "error", "text": "..." }` or `null`.

---

### `POST /api/architect/save`

Saves a new or updated network architecture. Also accepts an optional `input_type` field to bake model metadata (e.g. `ImageGrayscale`) into the model from design time, not just training time.

**Request body:**
```json
{
  "name": "my_model",
  "description": "optional description",
  "input_size": 784,
  "loss_type": "cross_entropy",
  "learning_rate": 0.01,
  "batch_size": 32,
  "epochs": 50,
  "layers": [
    { "neurons": 256, "activation": "relu" },
    { "neurons": 128, "activation": "relu" },
    { "neurons": 10,  "activation": "softmax" }
  ],
  "input_type": {
    "kind": "grayscale",
    "width": 28,
    "height": 28
  }
}
```

Valid `loss_type` values: `"mse"` | `"cross_entropy"` | `"bce"` | `"mae"` | `"huber"`

Valid `activation` values: `"sigmoid"` | `"relu"` | `"identity"` | `"softmax"` | `"tanh"` | `"leaky_relu"` | `"elu"` | `"gelu"` | `"swish"`

Valid `input_type.kind` values: `"numeric"` | `"grayscale"` | `"rgb"` (width + height required for image kinds)

**Validation rules:**
- A `softmax` output layer requires `loss_type = "cross_entropy"`
- A `sigmoid` output layer suggests `loss_type = "bce"`
- `input_size` must match the input dimension of the first layer

**Response:** `{ "ok": true }` or `{ "error": "..." }`

---

## Dataset

### `GET /api/dataset`

Returns the currently loaded dataset state.

**Response:**
```json
{
  "loaded": true,
  "source_name": "MNIST IDX",
  "feature_count": 784,
  "label_count": 10,
  "total_rows": 60000,
  "train_rows": 54000,
  "val_rows": 6000,
  "val_split_pct": 10,
  "preview_rows": [
    { "inputs": [0.0, 0.123, ...], "labels": [0.0, 0.0, 1.0, ...] }
  ],
  "tab_unlock": 7,
  "error": null
}
```

When no dataset is loaded, `loaded` is `false` and all other fields are `null`.

---

### `POST /api/dataset/upload`

Upload a CSV file as a dataset. The first row is treated as a header (ignored). All columns are numeric floats. The last N columns are labels; the rest are features.

**Request:** `multipart/form-data`
- Field `dataset` (file): CSV file
- Field `val_split` (text, optional): validation split percentage, 0–50 (default: 10)
- Field `label_cols` (text, optional): number of label columns from the right (default: 1)

**Response:** `DatasetResponse` (same shape as GET /api/dataset) with `error` set on failure.

---

### `POST /api/dataset/upload-idx`

Upload MNIST-format IDX binary files (images + labels).

**Request:** `multipart/form-data`
- Field `images` (file): IDX3 image file (e.g. `train-images-idx3-ubyte`)
- Field `labels` (file): IDX1 label file (e.g. `train-labels-idx1-ubyte`)
- Field `val_split` (text, optional): validation split percentage (default: 10)

Pixel values are normalized from `[0, 255]` to `[0.0, 1.0]`. Labels are one-hot encoded.

**Response:** `DatasetResponse` with `error` set on failure.

---

### `POST /api/dataset/builtin`

Load a built-in toy dataset.

**Request body:**
```json
{ "name": "xor", "val_split": 20 }
```

Valid `name` values: `"xor"` | `"circles"` | `"blobs"`

**Response:** `DatasetResponse` with `error` set on failure.

---

## Training

### `GET /api/train`

Returns current training status and configuration summary.

**Response:**
```json
{
  "status": "idle",
  "total_epochs": null,
  "model_path": null,
  "elapsed_total_ms": null,
  "was_stopped": null,
  "fail_reason": null,
  "epoch_history": [],
  "spec_name": null,
  "tab_unlock": 7
}
```

`status` values:
- `"idle"` — no training started
- `"running"` — training in progress; `total_epochs` is set
- `"done"` — training complete; `model_path`, `elapsed_total_ms`, `was_stopped` are set
- `"failed"` — training failed; `fail_reason` is set

`epoch_history` is an array of `EpochStats` objects (see SSE section for shape).

---

### `POST /api/train/start`

Starts a training run in a background thread. Requires an architecture spec and loaded dataset (otherwise returns an error).

**Request:** No body required.

**Response:** `{ "ok": true }` or `{ "error": "Set up architecture and dataset first." }`

Once started, connect to `GET /api/train/events` to receive real-time progress.

---

### `POST /api/train/stop`

Signals the running training job to stop after the current batch. The model is saved with whatever epochs have completed.

**Request:** No body required.

**Response:** `{ "ok": true }`

---

### `GET /api/train/events` — SSE Stream

Server-Sent Events stream for real-time training progress. Connect with `EventSource`. The stream stays open until training ends.

**Event types:**

```
event: epoch
data: {"epoch":1,"total_epochs":50,"train_loss":0.3241,"val_loss":0.3102,"train_accuracy":0.91,"val_accuracy":0.92,"elapsed_ms":843}

event: done
data: {"model_path":"trained_models/my_model.json","elapsed_total_ms":42000,"epochs_completed":50}

event: stopped
data: {"model_path":"trained_models/my_model.json","elapsed_total_ms":8000,"epoch_reached":10,"total_epochs":50}

event: failed
data: {"reason":"Layer size mismatch: ..."}
```

Reconnecting clients receive replayed `epoch` events from history before live events resume.

Keep-alive pings are sent as SSE comments every 15 seconds: `: ping`

**`EpochStats` shape:**
```json
{
  "epoch": 1,
  "total_epochs": 50,
  "train_loss": 0.3241,
  "val_loss": 0.3102,
  "train_accuracy": 0.9123,
  "val_accuracy": 0.9201,
  "elapsed_ms": 843
}
```

---

## Evaluate

### `GET /api/evaluate`

Returns the full epoch history and aggregate metrics for the most recent training run.

**Response:**
```json
{
  "epoch_history": [ /* array of EpochStats */ ],
  "best_train_loss": 0.0821,
  "best_val_loss": 0.0934,
  "best_train_accuracy": 0.9872,
  "best_val_accuracy": 0.9801,
  "confusion_matrix": [[980,0,1,...], [0,1130,2,...], ...],
  "class_labels": ["0","1","2","3","4","5","6","7","8","9"],
  "tab_unlock": 31
}
```

`confusion_matrix[i][j]` = number of samples with true class `i` predicted as class `j`.
All aggregate fields are `null` if no training has completed.

---

### `GET /api/evaluate/export`

Downloads epoch history as a CSV file.

**Response:** `Content-Type: text/csv`, `Content-Disposition: attachment; filename="epoch_history.csv"`

CSV columns: `epoch, train_loss, val_loss, train_accuracy, val_accuracy, elapsed_ms`

---

## Test / Inference

### `GET /api/test?model=NAME`

Returns the list of available models and metadata for the selected model.

**Query parameters:**
- `model` (optional): name of the model to select (stem of JSON filename, no extension)

**Response:**
```json
{
  "models": ["mnist", "first_network", "my_classifier"],
  "selected": "mnist",
  "model_info": {
    "name": "mnist",
    "input_type": { "type": "ImageGrayscale", "width": 28, "height": 28 },
    "output_labels": ["0","1","2","3","4","5","6","7","8","9"],
    "input_size": 784,
    "output_size": 10
  },
  "tab_unlock": 31
}
```

`model_info` is `null` when no model is selected or the model file cannot be loaded.

`input_type` values follow the `ferrite-nn` `InputType` enum serialization:
- `null` — no metadata; frontend should offer manual mode selection
- `{ "type": "Numeric" }` — numeric inputs only
- `{ "type": "ImageGrayscale", "width": 28, "height": 28 }` — grayscale image
- `{ "type": "ImageRgb", "width": 32, "height": 32 }` — RGB image

---

### `POST /api/test/infer`

Run inference on a model. Accepts multipart to support both numeric inputs and image file uploads.

**Request:** `multipart/form-data`
- `model` (text): model name
- `input_mode` (text): `"numeric"` | `"grayscale"` | `"rgb"`
- `input_width` (text, optional): target width for image resize (default: 28)
- `input_height` (text, optional): target height for image resize (default: 28)
- `inputs` (text, optional): comma-separated floats for numeric mode
- `image_file` (file, optional): PNG/JPEG/BMP/GIF for image modes

Image files are resized to `input_width × input_height` (Lanczos3 filter), converted to grayscale or RGB as appropriate, and normalized to `[0.0, 1.0]`.

**Response:**
```json
{
  "result_type": "softmax",
  "prediction": "7",
  "confidence": 0.9821,
  "all_scores": [
    { "label": "0", "score": 0.0012 },
    { "label": "7", "score": 0.9821 },
    ...
  ],
  "raw_values": null
}
```

`result_type` values:
- `"softmax"` — multi-class; `prediction` = highest-confidence class label, `confidence` = its score, `all_scores` = all classes sorted by score
- `"sigmoid"` — binary; `prediction` = `"true"` or `"false"`, `confidence` = sigmoid output value
- `"raw"` — all other activations; `raw_values` = raw output vector, `all_scores` = indexed entries

---

### `POST /api/test/import-model`

Upload a ferrite-nn JSON model file and add it to the local `trained_models/` directory.

**Request:** `multipart/form-data`
- `model_file` (file): `.json` model file exported from ferrite-nn

Validates that the JSON contains a `"layers"` key. The filename stem becomes the model name (sanitized to alphanumeric, `_`, `-`).

**Response:** `{ "ok": true, "name": "imported_model_name" }` or `{ "error": "..." }`

---

## Models

### `GET /api/models`

Lists all models in the `trained_models/` directory.

**Response:**
```json
[
  {
    "name": "mnist",
    "input_type": { "type": "ImageGrayscale", "width": 28, "height": 28 },
    "output_labels": ["0","1","2","3","4","5","6","7","8","9"]
  },
  {
    "name": "first_network",
    "input_type": null,
    "output_labels": null
  }
]
```

---

### `GET /api/models/:name/download`

Download a trained model as a JSON file.

**Path parameter:** `name` — model name (no `.json` extension)

**Response:** `Content-Type: application/json`, `Content-Disposition: attachment; filename="{name}.json"`

Returns 404 if the model file does not exist.

---

## Future: Auth headers (Phase 3)

When the FastAPI auth gateway is added in Phase 3, all `/api/*` requests will be routed through FastAPI first. The Rust service will trust a `X-User-Id` header injected by FastAPI after JWT validation. The response shapes defined here will not change — the auth layer is transparent to the frontend.

At that point, the frontend will also interact with FastAPI-only endpoints:
- `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`
- `POST /auth/refresh`
- `GET /auth/oauth/github`, `GET /auth/oauth/google` (and callbacks)
- `GET /auth/me`

These are documented separately in the `ferrite-studio` repository.
