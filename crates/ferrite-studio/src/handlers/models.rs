use axum::{
    extract::Path,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

use ferrite_nn::Network;

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ModelListEntry {
    pub name: String,
    pub input_type: Option<serde_json::Value>,
    pub output_labels: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// GET /api/models
// ---------------------------------------------------------------------------

pub async fn handle_list() -> Json<Vec<ModelListEntry>> {
    let dir = "trained_models";
    let entries = match std::fs::read_dir(dir) {
        Ok(e)  => e,
        Err(_) => return Json(vec![]),
    };

    let mut models: Vec<ModelListEntry> = entries.flatten()
        .filter_map(|e| {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                return None;
            }
            let name = path.file_stem()?.to_str()?.to_owned();

            let network = Network::load_json(path.to_str()?).ok();
            let input_type = network.as_ref()
                .and_then(|n| n.metadata.as_ref())
                .and_then(|m| m.input_type.as_ref())
                .and_then(|it| serde_json::to_value(it).ok());
            let output_labels = network.as_ref()
                .and_then(|n| n.metadata.as_ref())
                .and_then(|m| m.output_labels.clone());

            Some(ModelListEntry { name, input_type, output_labels })
        })
        .collect();

    models.sort_by(|a, b| a.name.cmp(&b.name));
    Json(models)
}

// ---------------------------------------------------------------------------
// GET /api/models/{name}/download
// ---------------------------------------------------------------------------

pub async fn handle_download(Path(name): Path<String>) -> Response {
    // Reject empty names or path traversal attempts.
    if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
        return (StatusCode::NOT_FOUND, "Not found").into_response();
    }

    let path = format!("trained_models/{}.json", name);
    match std::fs::read_to_string(&path) {
        Ok(json) => {
            let filename    = format!("{}.json", name);
            let disposition = format!("attachment; filename=\"{}\"", filename);
            let bytes       = json.into_bytes();
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE,        "application/json".to_owned()),
                    (header::CONTENT_DISPOSITION, disposition),
                ],
                bytes,
            )
                .into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}
