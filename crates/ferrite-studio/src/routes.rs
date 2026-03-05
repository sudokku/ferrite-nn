use std::sync::{Arc, Mutex};
use axum::{Router, routing::{get, post}, extract::DefaultBodyLimit};
use crate::state::StudioState;

/// Shared state passed to every handler via axum's `State` extractor.
/// `Arc<Mutex<StudioState>>` implements `Clone`, which axum requires for `State`.
pub type SharedState = Arc<Mutex<StudioState>>;

pub fn build_router(state: SharedState) -> Router {
    Router::new()
        // Architect
        .route("/api/architect",      get(crate::handlers::architect::handle_get))
        .route("/api/architect/save", post(crate::handlers::architect::handle_post))
        // Dataset
        .route("/api/dataset",            get(crate::handlers::dataset::handle_get))
        .route("/api/dataset/upload",     post(crate::handlers::dataset::handle_upload))
        .route("/api/dataset/upload-idx", post(crate::handlers::dataset::handle_upload_idx))
        .route("/api/dataset/builtin",    post(crate::handlers::dataset::handle_builtin))
        // Train
        .route("/api/train",        get(crate::handlers::train::handle_get))
        .route("/api/train/start",  post(crate::handlers::train::handle_start))
        .route("/api/train/stop",   post(crate::handlers::train::handle_stop))
        .route("/api/train/events", get(crate::handlers::train_sse::handle))
        // Evaluate
        .route("/api/evaluate",        get(crate::handlers::evaluate::handle_get))
        .route("/api/evaluate/export", get(crate::handlers::evaluate::handle_export))
        // Test / Inference
        .route("/api/test",              get(crate::handlers::test::handle_get))
        .route("/api/test/infer",        post(crate::handlers::test::handle_infer))
        .route("/api/test/import-model", post(crate::handlers::test::handle_import_model))
        // Model management
        .route("/api/models",                 get(crate::handlers::models::handle_list))
        .route("/api/models/:name/download",  get(crate::handlers::models::handle_download))
        .layer(DefaultBodyLimit::max(200 * 1024 * 1024)) // 200 MB — allows MNIST-sized IDX uploads
        .with_state(state)
}
