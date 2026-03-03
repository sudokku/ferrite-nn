use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::routes::SharedState;
use crate::state::{DatasetState, FlashMessage};
use crate::util::csv::{parse_csv, LabelMode, builtin_xor, builtin_circles, builtin_blobs};
use crate::util::idx::parse_idx_pair;

const MAX_CSV_BYTES: usize = 50 * 1024 * 1024;  // 50 MB
const MAX_IDX_BYTES: usize = 100 * 1024 * 1024; // 100 MB (MNIST train set is ~47 MB)

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct DatasetResponse {
    pub loaded: bool,
    pub source_name: Option<String>,
    pub feature_count: Option<usize>,
    pub label_count: Option<usize>,
    pub total_rows: Option<usize>,
    pub train_rows: Option<usize>,
    pub val_rows: Option<usize>,
    pub val_split_pct: Option<u8>,
    pub preview_rows: Option<Vec<PreviewRow>>,
    pub tab_unlock: u8,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct PreviewRow {
    pub inputs: Vec<f64>,
    pub labels: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct BuiltinRequest {
    pub name: String,      // "xor" | "circles" | "blobs"
    pub val_split: Option<u8>,
}

// ---------------------------------------------------------------------------
// GET /api/dataset
// ---------------------------------------------------------------------------

pub async fn handle_get(State(state): State<SharedState>) -> Json<DatasetResponse> {
    let st = state.lock().unwrap();
    let tab_unlock = st.tab_unlock_mask();
    let ds = st.dataset.clone();
    drop(st);

    Json(build_response(ds, tab_unlock, None))
}

// ---------------------------------------------------------------------------
// POST /api/dataset/upload   (CSV multipart)
// ---------------------------------------------------------------------------

pub async fn handle_upload(
    State(state): State<SharedState>,
    mut multipart: axum::extract::Multipart,
) -> Json<serde_json::Value> {
    // Collect all multipart fields.
    let mut csv_bytes: Option<Vec<u8>> = None;
    let mut val_split: u8 = 20;
    let mut label_mode_s = "class_index".to_owned();
    let mut n_classes: usize = 2;
    let mut n_label_cols: usize = 1;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_owned();
        match name.as_str() {
            "dataset" => {
                match field.bytes().await {
                    Ok(b) => {
                        if b.len() > MAX_CSV_BYTES {
                            return Json(serde_json::json!({"error": "File exceeds 50 MB limit."}));
                        }
                        csv_bytes = Some(b.to_vec());
                    }
                    Err(_) => return Json(serde_json::json!({"error": "Failed to read uploaded file."})),
                }
            }
            "val_split" => {
                if let Ok(text) = field.text().await {
                    val_split = text.trim().parse::<u8>().unwrap_or(20).min(50);
                }
            }
            "label_mode" => {
                if let Ok(text) = field.text().await {
                    label_mode_s = text.trim().to_owned();
                }
            }
            "n_classes" => {
                if let Ok(text) = field.text().await {
                    n_classes = text.trim().parse::<usize>().unwrap_or(2).max(2);
                }
            }
            "n_label_cols" => {
                if let Ok(text) = field.text().await {
                    n_label_cols = text.trim().parse::<usize>().unwrap_or(1).max(1);
                }
            }
            _ => { let _ = field.bytes().await; }
        }
    }

    let csv_bytes = match csv_bytes {
        Some(b) if !b.is_empty() => b,
        _ => return Json(serde_json::json!({"error": "No CSV file was uploaded."})),
    };

    let label_mode = if label_mode_s == "one_hot" {
        LabelMode::OneHot { n_label_cols }
    } else {
        LabelMode::ClassIndex { n_classes }
    };

    let (inputs, labels) = match parse_csv(&csv_bytes, label_mode) {
        Ok(r)  => r,
        Err(e) => return Json(serde_json::json!({"error": e.to_string()})),
    };

    // Validate feature count against spec.
    {
        let st = state.lock().unwrap();
        if let Some(spec) = &st.spec {
            let expected = spec.layers.first().map(|l| l.input_size).unwrap_or(0);
            if expected > 0 && inputs[0].len() != expected {
                let err = format!(
                    "Feature count mismatch: model expects {} inputs, CSV has {}.",
                    expected, inputs[0].len()
                );
                return Json(serde_json::json!({"error": err}));
            }
        }
    }

    let ds = build_dataset_state(inputs, labels, val_split, "CSV upload".to_owned());
    let tab_unlock = {
        let mut st = state.lock().unwrap();
        st.dataset = Some(ds.clone());
        st.flash   = Some(FlashMessage::success("Dataset loaded successfully."));
        st.tab_unlock_mask()
    };

    Json(serde_json::to_value(build_response(Some(ds), tab_unlock, None)).unwrap())
}

// ---------------------------------------------------------------------------
// POST /api/dataset/upload-idx   (IDX multipart)
// ---------------------------------------------------------------------------

pub async fn handle_upload_idx(
    State(state): State<SharedState>,
    mut multipart: axum::extract::Multipart,
) -> Json<serde_json::Value> {
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut label_bytes: Option<Vec<u8>> = None;
    let mut val_split: u8 = 10;
    let mut n_classes: usize = 10;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_owned();
        match name.as_str() {
            "images_file" => {
                match field.bytes().await {
                    Ok(b) => {
                        if b.len() > MAX_IDX_BYTES {
                            return Json(serde_json::json!({"error": "Upload exceeds 100 MB limit."}));
                        }
                        image_bytes = Some(b.to_vec());
                    }
                    Err(_) => return Json(serde_json::json!({"error": "Failed to read images file."})),
                }
            }
            "labels_file" => {
                match field.bytes().await {
                    Ok(b) => {
                        label_bytes = Some(b.to_vec());
                    }
                    Err(_) => return Json(serde_json::json!({"error": "Failed to read labels file."})),
                }
            }
            "val_split" => {
                if let Ok(text) = field.text().await {
                    val_split = text.trim().parse::<u8>().unwrap_or(10).min(50);
                }
            }
            "n_classes" => {
                if let Ok(text) = field.text().await {
                    n_classes = text.trim().parse::<usize>().unwrap_or(10).max(2);
                }
            }
            _ => { let _ = field.bytes().await; }
        }
    }

    let image_bytes = match image_bytes {
        Some(b) if !b.is_empty() => b,
        _ => return Json(serde_json::json!({"error": "No IDX image file was uploaded (field: images_file)."})),
    };
    let label_bytes = match label_bytes {
        Some(b) if !b.is_empty() => b,
        _ => return Json(serde_json::json!({"error": "No IDX label file was uploaded (field: labels_file)."})),
    };

    let (inputs, labels) = match parse_idx_pair(&image_bytes, &label_bytes, n_classes) {
        Ok(r)  => r,
        Err(e) => return Json(serde_json::json!({"error": e})),
    };

    // Validate feature count against spec.
    {
        let st = state.lock().unwrap();
        if let Some(spec) = &st.spec {
            let expected = spec.layers.first().map(|l| l.input_size).unwrap_or(0);
            if expected > 0 && !inputs.is_empty() && inputs[0].len() != expected {
                let err = format!(
                    "Feature count mismatch: model expects {} inputs, IDX images have {} pixels.",
                    expected, inputs[0].len()
                );
                return Json(serde_json::json!({"error": err}));
            }
        }
    }

    let source_name = format!(
        "IDX upload ({} samples, {}x{} px, {} classes)",
        inputs.len(),
        (inputs.first().map(|r| r.len()).unwrap_or(0) as f64).sqrt() as usize,
        (inputs.first().map(|r| r.len()).unwrap_or(0) as f64).sqrt() as usize,
        n_classes,
    );

    let ds = build_dataset_state(inputs, labels, val_split, source_name);
    let tab_unlock = {
        let mut st = state.lock().unwrap();
        st.dataset = Some(ds.clone());
        st.flash   = Some(FlashMessage::success("IDX dataset loaded successfully."));
        st.tab_unlock_mask()
    };

    Json(serde_json::to_value(build_response(Some(ds), tab_unlock, None)).unwrap())
}

// ---------------------------------------------------------------------------
// POST /api/dataset/builtin
// ---------------------------------------------------------------------------

pub async fn handle_builtin(
    State(state): State<SharedState>,
    Json(req): Json<BuiltinRequest>,
) -> Json<serde_json::Value> {
    let name = req.name.as_str();

    // XOR has only 4 samples — any validation split causes misleading metrics.
    let val_split: u8 = if name == "xor" {
        0
    } else {
        req.val_split.unwrap_or(20).min(50)
    };

    let (inputs, labels, source_name) = match name {
        "circles" => { let (i, l) = builtin_circles(200); (i, l, "Circles (200)".to_owned()) }
        "blobs"   => { let (i, l) = builtin_blobs(200);   (i, l, "Blobs (200)".to_owned())   }
        _         => { let (i, l) = builtin_xor();        (i, l, "XOR".to_owned())            }
    };

    // Validate feature count.
    {
        let st = state.lock().unwrap();
        if let Some(spec) = &st.spec {
            let expected = spec.layers.first().map(|l| l.input_size).unwrap_or(0);
            if expected > 0 && !inputs.is_empty() && inputs[0].len() != expected {
                let err = format!(
                    "Feature count mismatch: model expects {} inputs, '{}' has {}.",
                    expected, name, inputs[0].len()
                );
                return Json(serde_json::json!({"error": err}));
            }
        }
    }

    let ds = build_dataset_state(inputs, labels, val_split, source_name);
    let tab_unlock = {
        let mut st = state.lock().unwrap();
        st.dataset = Some(ds.clone());
        st.flash   = Some(FlashMessage::success("Dataset loaded successfully."));
        st.tab_unlock_mask()
    };

    Json(serde_json::to_value(build_response(Some(ds), tab_unlock, None)).unwrap())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_response(
    ds: Option<DatasetState>,
    tab_unlock: u8,
    error: Option<String>,
) -> DatasetResponse {
    match ds {
        None => DatasetResponse {
            loaded: false,
            source_name: None,
            feature_count: None,
            label_count: None,
            total_rows: None,
            train_rows: None,
            val_rows: None,
            val_split_pct: None,
            preview_rows: None,
            tab_unlock,
            error,
        },
        Some(d) => {
            let preview = d.preview_rows.iter()
                .map(|(inp, lbl)| PreviewRow { inputs: inp.clone(), labels: lbl.clone() })
                .collect();
            DatasetResponse {
                loaded: true,
                source_name: Some(d.source_name.clone()),
                feature_count: Some(d.feature_count),
                label_count: Some(d.label_count),
                total_rows: Some(d.total_rows),
                train_rows: Some(d.train_inputs.len()),
                val_rows: Some(d.val_inputs.len()),
                val_split_pct: Some(d.val_split_pct),
                preview_rows: Some(preview),
                tab_unlock,
                error,
            }
        }
    }
}

pub fn build_dataset_state(
    inputs: Vec<Vec<f64>>,
    labels: Vec<Vec<f64>>,
    val_split_pct: u8,
    source_name: String,
) -> DatasetState {
    let total = inputs.len();
    let feature_count = inputs.first().map(|r| r.len()).unwrap_or(0);
    let label_count   = labels.first().map(|r| r.len()).unwrap_or(0);

    let val_n   = (total * val_split_pct as usize) / 100;
    let train_n = total - val_n;

    let preview_rows: Vec<(Vec<f64>, Vec<f64>)> = inputs.iter().zip(labels.iter())
        .take(5)
        .map(|(i, l)| (i.clone(), l.clone()))
        .collect();

    let (train_inputs, val_inputs) = inputs.split_at(train_n);
    let (train_labels, val_labels) = labels.split_at(train_n);

    DatasetState {
        train_inputs:  train_inputs.to_vec(),
        train_labels:  train_labels.to_vec(),
        val_inputs:    val_inputs.to_vec(),
        val_labels:    val_labels.to_vec(),
        feature_count,
        label_count,
        total_rows: total,
        val_split_pct,
        source_name,
        preview_rows,
    }
}
