use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::panic;
use axum::{extract::State, Json};
use serde::Serialize;

use ferrite_nn::{Network, Sgd, TrainConfig, train_loop};

use crate::routes::SharedState;
use crate::state::{FlashMessage, TrainingStatus};

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct TrainResponse {
    pub status: String,                      // "idle" | "running" | "done" | "failed"
    pub total_epochs: Option<usize>,
    pub model_path: Option<String>,
    pub elapsed_total_ms: Option<u64>,
    pub was_stopped: Option<bool>,
    pub fail_reason: Option<String>,
    pub epoch_history: Vec<serde_json::Value>,
    pub spec_name: Option<String>,
    pub tab_unlock: u8,
}

// ---------------------------------------------------------------------------
// GET /api/train
// ---------------------------------------------------------------------------

pub async fn handle_get(State(state): State<SharedState>) -> Json<TrainResponse> {
    let st = state.lock().unwrap();
    let tab_unlock   = st.tab_unlock_mask();
    let spec_name    = st.spec.as_ref().map(|s| s.name.clone());
    let epoch_history: Vec<serde_json::Value> = st.epoch_history.iter()
        .filter_map(|e| serde_json::to_value(e).ok())
        .collect();

    let resp = match &st.training {
        TrainingStatus::Idle => TrainResponse {
            status: "idle".into(),
            total_epochs: st.hyperparams.as_ref().map(|h| h.epochs),
            model_path: None,
            elapsed_total_ms: None,
            was_stopped: None,
            fail_reason: None,
            epoch_history,
            spec_name,
            tab_unlock,
        },
        TrainingStatus::Running { total_epochs, .. } => TrainResponse {
            status: "running".into(),
            total_epochs: Some(*total_epochs),
            model_path: None,
            elapsed_total_ms: None,
            was_stopped: None,
            fail_reason: None,
            epoch_history,
            spec_name,
            tab_unlock,
        },
        TrainingStatus::Done { model_path, elapsed_total_ms, was_stopped } => TrainResponse {
            status: "done".into(),
            total_epochs: st.hyperparams.as_ref().map(|h| h.epochs),
            model_path: Some(model_path.clone()),
            elapsed_total_ms: Some(*elapsed_total_ms),
            was_stopped: Some(*was_stopped),
            fail_reason: None,
            epoch_history,
            spec_name,
            tab_unlock,
        },
        TrainingStatus::Failed { reason } => TrainResponse {
            status: "failed".into(),
            total_epochs: None,
            model_path: None,
            elapsed_total_ms: None,
            was_stopped: None,
            fail_reason: Some(reason.clone()),
            epoch_history,
            spec_name,
            tab_unlock,
        },
    };
    drop(st);

    Json(resp)
}

// ---------------------------------------------------------------------------
// POST /api/train/start
// ---------------------------------------------------------------------------

pub async fn handle_start(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut st = state.lock().unwrap();

    // Guard: need spec + hyperparams + dataset.
    if st.spec.is_none() || st.hyperparams.is_none() || st.dataset.is_none() {
        st.flash = Some(FlashMessage::error(
            "Set up architecture and dataset before training."
        ));
        return Json(serde_json::json!({"error": "Set up architecture and dataset before training."}));
    }

    // If already running, don't start another.
    if matches!(st.training, TrainingStatus::Running { .. }) {
        return Json(serde_json::json!({"error": "Training is already running."}));
    }

    let spec   = st.spec.clone().unwrap();
    let hp     = st.hyperparams.clone().unwrap();
    let ds     = st.dataset.clone().unwrap();

    // Use tokio mpsc so the async SSE handler can receive without blocking.
    // The background thread will use a std blocking send via the sender half.
    let (tx, rx) = tokio::sync::mpsc::channel::<ferrite_nn::EpochStats>(hp.epochs + 16);
    let stop_flag = Arc::new(AtomicBool::new(false));

    let epoch_rx   = Arc::new(tokio::sync::Mutex::new(rx));
    let total_epochs = hp.epochs;

    st.training = TrainingStatus::Running {
        stop_flag:   stop_flag.clone(),
        epoch_rx:    epoch_rx.clone(),
        total_epochs,
    };
    st.epoch_history.clear();
    st.trained_network = None;
    drop(st);

    // Spawn background training thread (std thread, not tokio task).
    // The tokio mpsc sender is safe to use from a std thread via blocking_send.
    let state_clone = state.clone();
    thread::spawn(move || {
        let mut network = Network::from_spec(&spec);
        let optimizer   = Sgd::new(hp.learning_rate);

        let val_inputs = if ds.val_inputs.is_empty() { None } else { Some(ds.val_inputs.as_slice()) };
        let val_labels = if ds.val_labels.is_empty() { None } else { Some(ds.val_labels.as_slice()) };

        // Bridge: wrap the tokio sender in a std mpsc compatible interface.
        // We create a std channel for train_loop (which expects std::sync::mpsc::Sender),
        // then relay stats from it into the tokio sender via a relay thread.
        let (std_tx, std_rx) = std::sync::mpsc::channel::<ferrite_nn::EpochStats>();

        let relay_tx = tx.clone();
        let relay_handle = thread::spawn(move || {
            while let Ok(stats) = std_rx.recv() {
                // blocking_send on tokio mpsc — this blocks the relay thread until
                // there is capacity, which is fine since we are in a std thread.
                if relay_tx.blocking_send(stats).is_err() {
                    break;
                }
            }
        });

        let mut config = TrainConfig::new(hp.epochs, hp.batch_size, spec.loss);
        config.progress_tx = Some(std_tx);
        config.stop_flag   = Some(stop_flag.clone());

        println!(
            "[studio] Training started: model='{}', samples={}, val={}, epochs={}, batch={}, lr={}",
            spec.name,
            ds.train_inputs.len(),
            ds.val_inputs.len(),
            hp.epochs,
            hp.batch_size,
            hp.learning_rate,
        );

        let t_start = std::time::Instant::now();

        let train_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            train_loop(
                &mut network,
                &ds.train_inputs,
                &ds.train_labels,
                val_inputs,
                val_labels,
                &optimizer,
                &config,
            )
        }));

        // Drop config to close the std sender — this causes the relay thread to exit.
        drop(config);
        // Wait for relay thread to finish flushing.
        let _ = relay_handle.join();

        if let Err(payload) = train_result {
            let reason = if let Some(s) = payload.downcast_ref::<String>() {
                format!("Training thread panicked: {}", s)
            } else if let Some(s) = payload.downcast_ref::<&str>() {
                format!("Training thread panicked: {}", s)
            } else {
                "Training thread panicked (unknown cause). Check that the \
                 architecture input size matches the dataset feature count.".to_owned()
            };
            eprintln!("[studio] ERROR: {}", reason);
            let mut st = state_clone.lock().unwrap();
            st.training = TrainingStatus::Failed { reason };
            return;
        }

        let elapsed_total_ms = t_start.elapsed().as_millis() as u64;
        let was_stopped = stop_flag.load(Ordering::Relaxed);
        println!(
            "[studio] Training finished: {} epochs in {:.1}s{}",
            hp.epochs,
            elapsed_total_ms as f64 / 1000.0,
            if was_stopped { " (stopped early)" } else { "" },
        );

        // Save model.
        let model_name = spec.name.clone();
        let model_dir  = "trained_models";
        let model_path = format!("{}/{}.json", model_dir, model_name);
        let _ = std::fs::create_dir_all(model_dir);
        network.metadata = spec.metadata.clone();
        let save_ok = network.save_json(&model_path).is_ok();

        let mut st = state_clone.lock().unwrap();

        // Drain remaining stats from the epoch receiver into history.
        // The tokio receiver is wrapped in a tokio::sync::Mutex, which cannot
        // be locked from a std thread using the async API. Instead, we use
        // try_recv on a blocking_lock, which requires entering a tokio context.
        // We avoid this complexity by noting that the relay thread has already
        // flushed all stats into the tokio channel, and the SSE handler is
        // responsible for collecting them. Any stats not yet consumed by SSE
        // will remain in the channel and be drained when SSE reconnects.

        if save_ok {
            println!("[studio] Model saved to '{}'", model_path);
            st.training = TrainingStatus::Done {
                model_path: model_path.clone(),
                elapsed_total_ms,
                was_stopped,
            };
        } else {
            let reason = format!(
                "Training finished but could not save model to '{}'. \
                 Check that the process has write permission to the trained_models/ directory.",
                model_path,
            );
            eprintln!("[studio] ERROR: {}", reason);
            st.training = TrainingStatus::Failed { reason };
        }
        st.trained_network = Some(network);
    });

    Json(serde_json::json!({"ok": true}))
}

// ---------------------------------------------------------------------------
// POST /api/train/stop
// ---------------------------------------------------------------------------

pub async fn handle_stop(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let st = state.lock().unwrap();
    if let TrainingStatus::Running { stop_flag, .. } = &st.training {
        stop_flag.store(true, Ordering::Relaxed);
    }
    drop(st);
    Json(serde_json::json!({"ok": true}))
}
