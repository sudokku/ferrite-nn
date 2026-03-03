/// ferrite-nn Studio
///
/// A full browser-based neural network creation, training, and testing platform.
/// Served by a synchronous tiny_http server; no JavaScript frameworks required.
///
/// Run with:
///   cargo run --bin studio --release
/// Then open http://127.0.0.1:7878
///
/// Tabs:
///   1. Architect — define network layers, loss, and hyperparameters
///   2. Dataset   — upload a CSV or pick a built-in toy dataset
///   3. Train     — train with real-time SSE loss chart
///   4. Evaluate  — loss curve, metrics table, confusion matrix
///   5. Test      — run inference on any saved model

mod state;
mod render;
mod routes;
mod handlers;
mod util;

use std::sync::{Arc, Mutex};
use tiny_http::Server;

use state::StudioState;

fn main() {
    let addr = "127.0.0.1:7878";
    let server = Server::http(addr).expect("Failed to bind HTTP server");

    let shared_state = Arc::new(Mutex::new(StudioState::new()));

    println!("╔══════════════════════════════════════════════╗");
    println!("║          ferrite-nn Studio                   ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║  Open in your browser:                       ║");
    println!("║  http://{}                 ║", addr);
    println!("╠══════════════════════════════════════════════╣");
    println!("║  Tabs: Architect > Dataset > Train >         ║");
    println!("║        Evaluate > Test                       ║");
    println!("╚══════════════════════════════════════════════╝");

    // Ensure trained_models/ directory exists.
    let _ = std::fs::create_dir_all("trained_models");

    // Each request is dispatched on its own thread so the SSE handler
    // (which blocks for the entire training duration) does not stall
    // regular page loads and form submissions.
    for request in server.incoming_requests() {
        let state_clone = shared_state.clone();
        std::thread::spawn(move || {
            routes::dispatch(request, state_clone);
        });
    }
}
