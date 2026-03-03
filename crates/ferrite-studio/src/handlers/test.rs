use axum::{
    extract::{Query, State},
    Json,
};
use serde::Serialize;
use std::collections::HashMap;

use ferrite_nn::{ActivationFunction, Network};

use crate::routes::SharedState;
use crate::util::image::{image_bytes_to_grayscale_input, image_bytes_to_rgb_input};

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct TestResponse {
    pub models: Vec<String>,
    pub selected: Option<String>,
    pub model_info: Option<ModelInfo>,
    pub tab_unlock: u8,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub input_type: Option<serde_json::Value>,  // serialized InputType or null
    pub output_labels: Option<Vec<String>>,
    pub input_size: usize,
    pub output_size: usize,
}

#[derive(Serialize)]
pub struct InferResult {
    pub result_type: String,          // "softmax" | "sigmoid" | "raw"
    pub prediction: Option<String>,
    pub confidence: Option<f64>,
    pub all_scores: Vec<ScoreEntry>,
    pub raw_values: Option<Vec<f64>>,
}

#[derive(Serialize)]
pub struct ScoreEntry {
    pub label: String,
    pub score: f64,
}

// ---------------------------------------------------------------------------
// GET /api/test?model=NAME
// ---------------------------------------------------------------------------

pub async fn handle_get(
    State(state): State<SharedState>,
    Query(params): Query<HashMap<String, String>>,
) -> Json<TestResponse> {
    let st = state.lock().unwrap();
    let tab_unlock = st.tab_unlock_mask();
    drop(st);

    let selected = params.get("model").cloned().filter(|s| !s.is_empty());
    let models   = list_models();

    let model_info = selected.as_deref().and_then(|name| {
        load_model_info(name)
    });

    Json(TestResponse {
        models,
        selected,
        model_info,
        tab_unlock,
    })
}

// ---------------------------------------------------------------------------
// POST /api/test/infer   (multipart)
// ---------------------------------------------------------------------------

pub async fn handle_infer(
    State(_state): State<SharedState>,
    mut multipart: axum::extract::Multipart,
) -> Json<serde_json::Value> {

    // Collect multipart fields.
    let mut model_name  = String::new();
    let mut input_mode  = "numeric".to_owned();
    let mut width: u32  = 28;
    let mut height: u32 = 28;
    let mut inputs_text = String::new();
    let mut image_bytes: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_owned();
        match name.as_str() {
            "model" => {
                if let Ok(t) = field.text().await { model_name = t; }
            }
            "input_mode" => {
                if let Ok(t) = field.text().await { input_mode = t; }
            }
            "input_width" => {
                if let Ok(t) = field.text().await {
                    width = t.trim().parse().unwrap_or(28);
                }
            }
            "input_height" => {
                if let Ok(t) = field.text().await {
                    height = t.trim().parse().unwrap_or(28);
                }
            }
            "inputs" => {
                if let Ok(t) = field.text().await { inputs_text = t; }
            }
            "image_file" => {
                if let Ok(b) = field.bytes().await {
                    if !b.is_empty() {
                        image_bytes = Some(b.to_vec());
                    }
                }
            }
            _ => { let _ = field.bytes().await; }
        }
    }

    let result = match input_mode.as_str() {
        "grayscale" => match image_bytes {
            Some(bytes) => run_infer_grayscale(&model_name, &bytes, width, height),
            None        => Err("No image file was uploaded.".to_owned()),
        },
        "rgb" => match image_bytes {
            Some(bytes) => run_infer_rgb(&model_name, &bytes, width, height),
            None        => Err("No image file was uploaded.".to_owned()),
        },
        _ => run_infer_numeric(&model_name, &inputs_text),
    };

    match result {
        Ok(infer) => Json(serde_json::to_value(infer).unwrap_or(serde_json::json!({}))),
        Err(e)    => Json(serde_json::json!({"error": e})),
    }
}

// ---------------------------------------------------------------------------
// POST /api/test/import-model   (multipart)
// ---------------------------------------------------------------------------

pub async fn handle_import_model(
    State(_state): State<SharedState>,
    mut multipart: axum::extract::Multipart,
) -> Json<serde_json::Value> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut raw_filename = "imported_model".to_owned();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name     = field.name().unwrap_or("").to_owned();
        let filename = field.file_name().unwrap_or("").to_owned();
        if name == "model_file" {
            if !filename.is_empty() {
                raw_filename = filename;
            }
            match field.bytes().await {
                Ok(b) if !b.is_empty() => { file_bytes = Some(b.to_vec()); }
                _ => {}
            }
        } else {
            let _ = field.bytes().await;
        }
    }

    let file_bytes = match file_bytes {
        Some(b) => b,
        None    => return Json(serde_json::json!({"error": "No JSON file was uploaded."})),
    };

    // Basic validation: must be valid JSON with a "layers" key.
    let json_val: serde_json::Value = match serde_json::from_slice(&file_bytes) {
        Ok(v)  => v,
        Err(_) => return Json(serde_json::json!({"error": "Uploaded file is not valid JSON."})),
    };
    if json_val.get("layers").is_none() {
        return Json(serde_json::json!({
            "error": "JSON does not appear to be a Ferrite model (missing \"layers\" field)."
        }));
    }

    // Sanitize the filename stem.
    let stem = std::path::Path::new(&raw_filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("imported_model");
    let sanitized: String = stem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    let model_name = if sanitized.is_empty() { "imported_model".to_owned() } else { sanitized };

    let model_dir  = "trained_models";
    let model_path = format!("{}/{}.json", model_dir, model_name);

    if let Err(_) = std::fs::create_dir_all(model_dir) {
        return Json(serde_json::json!({"error": "Could not create trained_models/ directory."}));
    }
    if let Err(_) = std::fs::write(&model_path, &file_bytes) {
        return Json(serde_json::json!({
            "error": format!("Could not write model to '{}'.", model_path)
        }));
    }

    Json(serde_json::json!({"ok": true, "name": model_name}))
}

// ---------------------------------------------------------------------------
// Helpers — model listing and info
// ---------------------------------------------------------------------------

pub fn list_models() -> Vec<String> {
    let dir = "trained_models";
    match std::fs::read_dir(dir) {
        Ok(entries) => {
            let mut names: Vec<String> = entries.flatten()
                .filter_map(|e| {
                    let path = e.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json") {
                        path.file_stem().and_then(|s| s.to_str()).map(|s| s.to_owned())
                    } else {
                        None
                    }
                })
                .collect();
            names.sort();
            names
        }
        Err(_) => vec![],
    }
}

fn load_model_info(name: &str) -> Option<ModelInfo> {
    if name.is_empty() { return None; }
    let path = format!("trained_models/{}.json", name);
    let network = Network::load_json(&path).ok()?;
    let input_size  = network.input_size();
    let output_size = network.layers.last().map(|l| l.size).unwrap_or(0);
    let input_type  = network.metadata.as_ref()
        .and_then(|m| m.input_type.as_ref())
        .and_then(|it| serde_json::to_value(it).ok());
    let output_labels = network.metadata.as_ref()
        .and_then(|m| m.output_labels.clone());

    Some(ModelInfo {
        name: name.to_owned(),
        input_type,
        output_labels,
        input_size,
        output_size,
    })
}

// ---------------------------------------------------------------------------
// Inference runners
// ---------------------------------------------------------------------------

fn run_infer_numeric(model_name: &str, raw_inputs: &str) -> Result<InferResult, String> {
    let path = format!("trained_models/{}.json", model_name);
    let mut network = Network::load_json(&path)
        .map_err(|e| format!("Could not load model '{}': {}", model_name, e))?;

    if network.layers.is_empty() {
        return Err("Model has no layers.".into());
    }

    let inputs: Vec<f64> = raw_inputs
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();

    let expected = network.input_size();
    if inputs.len() != expected {
        return Err(format!(
            "Input length mismatch: model expects {} values, got {}.",
            expected, inputs.len()
        ));
    }

    let output = network.forward(inputs);
    let labels = network.metadata.as_ref().and_then(|m| m.output_labels.as_deref());
    let activation = network.output_activation()
        .cloned()
        .unwrap_or(ActivationFunction::Identity);
    Ok(format_infer_result(&output, labels, &activation))
}

fn run_infer_grayscale(
    model_name: &str,
    image_bytes: &[u8],
    width: u32,
    height: u32,
) -> Result<InferResult, String> {
    let path = format!("trained_models/{}.json", model_name);
    let mut network = Network::load_json(&path)
        .map_err(|e| format!("Could not load model '{}': {}", model_name, e))?;

    if network.layers.is_empty() {
        return Err("Model has no layers.".into());
    }

    let inputs = image_bytes_to_grayscale_input(image_bytes, width, height)
        .map_err(|e| format!("Image decode error: {}", e))?;

    let output = network.forward(inputs);
    let labels = network.metadata.as_ref().and_then(|m| m.output_labels.as_deref());
    let activation = network.output_activation()
        .cloned()
        .unwrap_or(ActivationFunction::Identity);
    Ok(format_infer_result(&output, labels, &activation))
}

fn run_infer_rgb(
    model_name: &str,
    image_bytes: &[u8],
    width: u32,
    height: u32,
) -> Result<InferResult, String> {
    let path = format!("trained_models/{}.json", model_name);
    let mut network = Network::load_json(&path)
        .map_err(|e| format!("Could not load model '{}': {}", model_name, e))?;

    if network.layers.is_empty() {
        return Err("Model has no layers.".into());
    }

    let inputs = image_bytes_to_rgb_input(image_bytes, width, height)
        .map_err(|e| format!("Image decode error: {}", e))?;

    let output = network.forward(inputs);
    let labels = network.metadata.as_ref().and_then(|m| m.output_labels.as_deref());
    let activation = network.output_activation()
        .cloned()
        .unwrap_or(ActivationFunction::Identity);
    Ok(format_infer_result(&output, labels, &activation))
}

// ---------------------------------------------------------------------------
// Output formatters
// ---------------------------------------------------------------------------

fn format_infer_result(
    output: &[f64],
    labels: Option<&[String]>,
    activation: &ActivationFunction,
) -> InferResult {
    match activation {
        ActivationFunction::Softmax => format_softmax(output, labels),
        ActivationFunction::Sigmoid if output.len() == 1 => format_sigmoid(output[0]),
        _ => format_raw(output),
    }
}

fn format_softmax(output: &[f64], labels: Option<&[String]>) -> InferResult {
    let n = output.len();

    let label_for = |i: usize| -> String {
        labels.and_then(|l| l.get(i)).cloned().unwrap_or_else(|| i.to_string())
    };

    let (best, best_conf) = output.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));

    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| output[b].partial_cmp(&output[a]).unwrap());

    let all_scores: Vec<ScoreEntry> = sorted.iter().map(|&i| ScoreEntry {
        label: label_for(i),
        score: output[i],
    }).collect();

    InferResult {
        result_type: "softmax".into(),
        prediction: Some(label_for(best)),
        confidence: Some(best_conf),
        all_scores,
        raw_values: None,
    }
}

fn format_sigmoid(value: f64) -> InferResult {
    InferResult {
        result_type: "sigmoid".into(),
        prediction: Some(format!("{:.4}", value)),
        confidence: Some(value),
        all_scores: vec![
            ScoreEntry { label: "output".into(), score: value },
        ],
        raw_values: Some(vec![value]),
    }
}

fn format_raw(output: &[f64]) -> InferResult {
    let all_scores: Vec<ScoreEntry> = output.iter().enumerate()
        .map(|(i, &v)| ScoreEntry { label: i.to_string(), score: v })
        .collect();

    InferResult {
        result_type: "raw".into(),
        prediction: None,
        confidence: None,
        all_scores,
        raw_values: Some(output.to_vec()),
    }
}
