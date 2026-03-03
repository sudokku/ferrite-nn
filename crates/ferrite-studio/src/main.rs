/// ferrite-nn Studio
///
/// A full browser-based neural network creation, training, and testing platform.
/// Served by an axum HTTP server exposing a JSON REST API.
///
/// Run with:
///   cargo run --bin studio --release
/// Then point your API client or frontend at http://127.0.0.1:7878

mod state;
mod routes;
mod handlers;
mod util;

use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

use state::StudioState;

#[tokio::main]
async fn main() {
    let addr = "127.0.0.1:7878";
    let shared_state = Arc::new(Mutex::new(StudioState::new()));

    println!("╔══════════════════════════════════════════════╗");
    println!("║          ferrite-nn Studio (API)             ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║  API available at:                           ║");
    println!("║  http://{}                 ║", addr);
    println!("╚══════════════════════════════════════════════╝");

    let _ = std::fs::create_dir_all("trained_models");

    let app = routes::build_router(shared_state)
        .layer(CorsLayer::permissive()); // permissive for local dev; tighten in production

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
