use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

use crate::routes::SharedState;

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct EvaluateResponse {
    pub epoch_history: Vec<serde_json::Value>,
    pub best_train_loss: Option<f64>,
    pub best_val_loss: Option<f64>,
    pub best_train_accuracy: Option<f64>,
    pub best_val_accuracy: Option<f64>,
    pub confusion_matrix: Option<Vec<Vec<usize>>>,
    pub class_labels: Option<Vec<String>>,
    pub tab_unlock: u8,
}

// ---------------------------------------------------------------------------
// GET /api/evaluate
// ---------------------------------------------------------------------------

pub async fn handle_get(State(state): State<SharedState>) -> Json<EvaluateResponse> {
    let st = state.lock().unwrap();
    let tab_unlock = st.tab_unlock_mask();
    let history    = st.epoch_history.clone();

    let epoch_history: Vec<serde_json::Value> = history.iter()
        .filter_map(|e| serde_json::to_value(e).ok())
        .collect();

    // Best metrics over all epochs.
    let best_train_loss     = history.iter().map(|e| e.train_loss).reduce(f64::min);
    let best_val_loss       = history.iter().filter_map(|e| e.val_loss).reduce(f64::min);
    let best_train_accuracy = history.iter().filter_map(|e| e.train_accuracy).reduce(f64::max);
    let best_val_accuracy   = history.iter().filter_map(|e| e.val_accuracy).reduce(f64::max);

    // Confusion matrix from trained network on validation set.
    let (confusion_matrix, class_labels) = if let (Some(network_ref), Some(ds)) =
        (&st.trained_network, &st.dataset)
    {
        if !ds.val_inputs.is_empty() {
            let matrix  = build_confusion_matrix(network_ref, &ds.val_inputs, &ds.val_labels);
            // Use numeric labels "0", "1", ... unless model has output_labels.
            let n_classes = ds.val_labels.first().map(|l| l.len()).unwrap_or(0);
            let labels: Option<Vec<String>> = network_ref.metadata
                .as_ref()
                .and_then(|m| m.output_labels.clone());
            let class_labels = labels.or_else(|| {
                if n_classes > 0 {
                    Some((0..n_classes).map(|i| i.to_string()).collect())
                } else {
                    None
                }
            });
            (matrix, class_labels)
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    drop(st);

    Json(EvaluateResponse {
        epoch_history,
        best_train_loss,
        best_val_loss,
        best_train_accuracy,
        best_val_accuracy,
        confusion_matrix,
        class_labels,
        tab_unlock,
    })
}

// ---------------------------------------------------------------------------
// GET /api/evaluate/export
// ---------------------------------------------------------------------------

pub async fn handle_export(State(state): State<SharedState>) -> Response {
    let st      = state.lock().unwrap();
    let history = st.epoch_history.clone();
    drop(st);

    let json = serde_json::to_string_pretty(&history).unwrap_or_else(|_| "[]".into());
    let bytes = json.into_bytes();

    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE,        "application/json".to_owned()),
            (header::CONTENT_DISPOSITION, "attachment; filename=\"epoch_history.json\"".to_owned()),
        ],
        bytes,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Confusion matrix helper
// ---------------------------------------------------------------------------

fn build_confusion_matrix(
    network: &ferrite_nn::Network,
    val_inputs: &[Vec<f64>],
    val_labels: &[Vec<f64>],
) -> Option<Vec<Vec<usize>>> {
    if val_labels.is_empty() { return None; }

    let n_classes = val_labels[0].len();
    if n_classes < 2 { return None; }

    let mut matrix = vec![vec![0usize; n_classes]; n_classes];

    for (input, label) in val_inputs.iter().zip(val_labels.iter()) {
        let output    = network.forward(input);
        let predicted = argmax(&output);
        let truth     = argmax(label);
        if predicted < n_classes && truth < n_classes {
            matrix[truth][predicted] += 1;
        }
    }

    Some(matrix)
}

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
