use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use ferrite_nn::{ActivationFunction, InputType, LossType, ModelMetadata, NetworkSpec, LayerSpec};

use crate::state::{FlashKind, FlashMessage, Hyperparams, TrainingStatus};
use crate::routes::SharedState;

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ArchitectResponse {
    pub spec: Option<serde_json::Value>,
    pub hyperparams: Option<HyperparamsJson>,
    pub tab_unlock: u8,
    pub flash: Option<FlashJson>,
}

#[derive(Serialize)]
pub struct HyperparamsJson {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
}

#[derive(Serialize)]
pub struct FlashJson {
    pub kind: String, // "success" | "error"
    pub text: String,
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SaveArchitectRequest {
    pub name: String,
    pub description: Option<String>,
    pub input_size: usize,
    pub loss_type: String,       // "mse" | "cross_entropy" | "bce" | "mae" | "huber"
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub layers: Vec<LayerRequest>,
    pub input_type: Option<InputTypeRequest>,
}

#[derive(Deserialize)]
pub struct LayerRequest {
    pub neurons: usize,
    pub activation: String,
}

#[derive(Deserialize)]
pub struct InputTypeRequest {
    pub kind: String,          // "numeric" | "grayscale" | "rgb"
    pub width: Option<u32>,
    pub height: Option<u32>,
}

// ---------------------------------------------------------------------------
// GET /api/architect
// ---------------------------------------------------------------------------

pub async fn handle_get(State(state): State<SharedState>) -> Json<ArchitectResponse> {
    let mut st = state.lock().unwrap();
    let flash = st.take_flash();
    let tab_unlock = st.tab_unlock_mask();
    let spec = st.spec.clone();
    let hyperparams = st.hyperparams.clone();
    drop(st);

    let spec_json = spec.as_ref().and_then(|s| serde_json::to_value(s).ok());

    let hp_json = hyperparams.as_ref().map(|h| HyperparamsJson {
        learning_rate: h.learning_rate,
        batch_size: h.batch_size,
        epochs: h.epochs,
    });

    let flash_json = flash.map(|f| FlashJson {
        kind: match f.kind {
            FlashKind::Success => "success".into(),
            FlashKind::Error   => "error".into(),
        },
        text: f.text,
    });

    Json(ArchitectResponse {
        spec: spec_json,
        hyperparams: hp_json,
        tab_unlock,
        flash: flash_json,
    })
}

// ---------------------------------------------------------------------------
// POST /api/architect/save
// ---------------------------------------------------------------------------

pub async fn handle_post(
    State(state): State<SharedState>,
    Json(req): Json<SaveArchitectRequest>,
) -> Json<serde_json::Value> {
    let name = req.name.trim().to_owned();

    if name.is_empty() {
        return Json(serde_json::json!({"error": "Model name must not be empty."}));
    }

    if req.input_size == 0 {
        return Json(serde_json::json!({"error": "Input size must be a positive integer."}));
    }

    if req.learning_rate <= 0.0 {
        return Json(serde_json::json!({"error": "Learning rate must be a positive number."}));
    }

    if req.batch_size == 0 {
        return Json(serde_json::json!({"error": "Batch size must be a positive integer."}));
    }

    if req.epochs == 0 {
        return Json(serde_json::json!({"error": "Epochs must be a positive integer."}));
    }

    if req.layers.is_empty() {
        return Json(serde_json::json!({"error": "Add at least one layer."}));
    }

    for rl in &req.layers {
        if rl.neurons == 0 {
            return Json(serde_json::json!({"error": "Each layer must have at least 1 neuron."}));
        }
    }

    // Build LayerSpec list.
    let mut layer_specs: Vec<LayerSpec> = Vec::new();
    let mut prev_size = req.input_size;
    for rl in &req.layers {
        let activation = parse_activation(&rl.activation);
        layer_specs.push(LayerSpec { size: rl.neurons, input_size: prev_size, activation });
        prev_size = rl.neurons;
    }

    let loss = match req.loss_type.as_str() {
        "cross_entropy" => LossType::CrossEntropy,
        "bce"           => LossType::BinaryCrossEntropy,
        "mae"           => LossType::Mae,
        "huber"         => LossType::Huber,
        _               => LossType::Mse,
    };

    // Enforce Softmax <-> CrossEntropy consistency.
    let last_act = &layer_specs.last().unwrap().activation;
    if *last_act == ActivationFunction::Softmax && loss != LossType::CrossEntropy {
        return Json(serde_json::json!({
            "error": "Softmax output requires Cross-Entropy loss. Please change the loss function."
        }));
    }
    if *last_act != ActivationFunction::Softmax && loss == LossType::CrossEntropy {
        return Json(serde_json::json!({
            "error": "Cross-Entropy loss requires a Softmax output layer."
        }));
    }
    if *last_act == ActivationFunction::Softmax && loss == LossType::BinaryCrossEntropy {
        return Json(serde_json::json!({
            "error": "Binary Cross-Entropy loss must not be paired with a Softmax output. Use Sigmoid instead."
        }));
    }

    // Build metadata — include description if provided, and input_type if specified.
    let description = req.description.as_deref().unwrap_or("").trim().to_owned();
    let input_type = req.input_type.as_ref().and_then(|it| parse_input_type(it));

    let has_metadata = !description.is_empty() || input_type.is_some();
    let metadata = if has_metadata {
        Some(ModelMetadata {
            description: if description.is_empty() { None } else { Some(description) },
            input_type,
            output_labels: None,
        })
    } else {
        None
    };

    let mut spec = NetworkSpec { name: name.clone(), layers: layer_specs, loss, metadata };

    // If a description was provided and no metadata existed yet, ensure metadata is set.
    if !req.description.as_deref().unwrap_or("").trim().is_empty() && spec.metadata.is_none() {
        spec.metadata = Some(ModelMetadata {
            description: req.description.clone().map(|d| d.trim().to_owned()).filter(|d| !d.is_empty()),
            input_type: None,
            output_labels: None,
        });
    }

    let hyperparams = Hyperparams {
        learning_rate: req.learning_rate,
        batch_size: req.batch_size,
        epochs: req.epochs,
    };

    let mut st = state.lock().unwrap();
    st.spec        = Some(spec);
    st.hyperparams = Some(hyperparams);
    // Clear stale state when the architecture changes.
    st.dataset         = None;
    st.epoch_history.clear();
    st.trained_network = None;
    st.training        = TrainingStatus::Idle;
    st.flash = Some(FlashMessage::success(
        format!("Architecture '{}' saved successfully.", name)
    ));
    drop(st);

    Json(serde_json::json!({"ok": true}))
}

// ---------------------------------------------------------------------------
// Shared helpers (also used by other handlers)
// ---------------------------------------------------------------------------

pub fn parse_activation(s: &str) -> ActivationFunction {
    match s {
        "relu"       => ActivationFunction::ReLU,
        "softmax"    => ActivationFunction::Softmax,
        "identity"   => ActivationFunction::Identity,
        "tanh"       => ActivationFunction::Tanh,
        "leaky_relu" => ActivationFunction::LeakyReLU { alpha: 0.01 },
        "elu"        => ActivationFunction::Elu { alpha: 1.0 },
        "gelu"       => ActivationFunction::Gelu,
        "swish"      => ActivationFunction::Swish,
        _            => ActivationFunction::Sigmoid,
    }
}

fn parse_input_type(it: &InputTypeRequest) -> Option<InputType> {
    match it.kind.as_str() {
        "grayscale" => Some(InputType::ImageGrayscale {
            width:  it.width.unwrap_or(28),
            height: it.height.unwrap_or(28),
        }),
        "rgb" => Some(InputType::ImageRgb {
            width:  it.width.unwrap_or(28),
            height: it.height.unwrap_or(28),
        }),
        "numeric" => Some(InputType::Numeric),
        _ => None,
    }
}
