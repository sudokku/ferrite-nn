use std::io::Write;
use std::time::Duration;
use tiny_http::Request;

use crate::state::{SharedState, TrainingStatus};

/// `GET /train/events` — Server-Sent Events handler.
///
/// This handler consumes `request` (takes ownership so we can call
/// `into_writer`) and drives a long-lived loop that:
/// 1. Tries to receive an `EpochStats` from the training channel with a
///    500 ms timeout.
/// 2. On success — serializes the stats and writes an `event: epoch\n\n` frame.
/// 3. On timeout — writes a keep-alive `: ping\n\n` comment.
/// 4. On channel disconnect (training finished) — writes a `done` or `stopped`
///    event, then closes.
///
/// Client reconnection is handled natively by `EventSource`.
pub fn handle(request: Request, state: SharedState) {
    // tiny_http's `into_writer()` gives us the raw TCP stream so we can
    // write the HTTP response and then stream SSE frames directly.
    let mut writer = request.into_writer();

    // Write HTTP response headers manually (tiny_http into_writer path).
    let header = "HTTP/1.1 200 OK\r\n\
                  Content-Type: text/event-stream\r\n\
                  Cache-Control: no-cache\r\n\
                  Connection: keep-alive\r\n\
                  X-Accel-Buffering: no\r\n\
                  \r\n";
    if let Err(_) = write_all(&mut writer, header.as_bytes()) {
        return;
    }

    // Extract the receiver Arc from state (clone it out so we don't hold the lock).
    let epoch_rx = {
        let st = state.lock().unwrap();
        match &st.training {
            TrainingStatus::Running { epoch_rx, .. } => Some(epoch_rx.clone()),
            _ => None,
        }
    };

    let rx_arc = match epoch_rx {
        Some(r) => r,
        None    => {
            // Training is not Running — emit an event matching the actual state.
            let msg = {
                let st = state.lock().unwrap();
                match &st.training {
                    TrainingStatus::Done { model_path, elapsed_total_ms, was_stopped } => {
                        let ep    = st.epoch_history.len();
                        let total = st.hyperparams.as_ref().map(|h| h.epochs).unwrap_or(0);
                        if *was_stopped {
                            format!(
                                "event: stopped\ndata: {{\"model_path\":\"{mp}\",\"elapsed_total_ms\":{el},\"epoch_reached\":{ep},\"total_epochs\":{total}}}\n\n",
                                mp=model_path, el=elapsed_total_ms, ep=ep, total=total,
                            )
                        } else {
                            format!(
                                "event: done\ndata: {{\"model_path\":\"{mp}\",\"elapsed_total_ms\":{el},\"epochs_completed\":{ep}}}\n\n",
                                mp=model_path, el=elapsed_total_ms, ep=ep,
                            )
                        }
                    }
                    TrainingStatus::Failed { reason } => {
                        format!(
                            "event: failed\ndata: {{\"reason\":\"{}\"}}\n\n",
                            reason.replace('"', "\\\""),
                        )
                    }
                    _ => String::new(), // Idle — close without event
                }
            };
            if !msg.is_empty() {
                let _ = write_all(&mut writer, msg.as_bytes());
            }
            return;
        }
    };

    // Collect history so far from state and replay it immediately.
    {
        let st = state.lock().unwrap();
        for stats in &st.epoch_history {
            if let Ok(json) = serde_json::to_string(stats) {
                let msg = format!("event: epoch\ndata: {}\n\n", json);
                if write_all(&mut writer, msg.as_bytes()).is_err() { return; }
            }
        }
    }

    // Main receive loop.
    loop {
        let result = {
            let rx = rx_arc.lock().unwrap();
            rx.recv_timeout(Duration::from_millis(500))
        };

        match result {
            Ok(stats) => {
                // Push to epoch_history.
                {
                    let mut st = state.lock().unwrap();
                    st.epoch_history.push(stats.clone());
                }

                match serde_json::to_string(&stats) {
                    Ok(json) => {
                        let msg = format!("event: epoch\ndata: {}\n\n", json);
                        if write_all(&mut writer, msg.as_bytes()).is_err() { return; }
                    }
                    Err(_) => continue,
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Keep-alive ping.
                if write_all(&mut writer, b": ping\n\n").is_err() { return; }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // Training thread closed the sender — check final status.
                let training_status_json = {
                    let st = state.lock().unwrap();
                    match &st.training {
                        TrainingStatus::Done { model_path, elapsed_total_ms, was_stopped } => {
                            let ep    = st.epoch_history.len();
                            let total = st.hyperparams.as_ref().map(|h| h.epochs).unwrap_or(0);
                            if *was_stopped {
                                // User stopped training; model still saved — emit stopped event
                                // with the model path so the client can persist it.
                                format!(
                                    "event: stopped\ndata: {{\"model_path\":\"{mp}\",\"elapsed_total_ms\":{el},\"epoch_reached\":{ep},\"total_epochs\":{total}}}\n\n",
                                    mp    = model_path,
                                    el    = elapsed_total_ms,
                                    ep    = ep,
                                    total = total,
                                )
                            } else {
                                format!(
                                    "event: done\ndata: {{\"model_path\":\"{mp}\",\"elapsed_total_ms\":{el},\"epochs_completed\":{ep}}}\n\n",
                                    mp = model_path,
                                    el = elapsed_total_ms,
                                    ep = ep,
                                )
                            }
                        }
                        TrainingStatus::Failed { reason } => {
                            format!(
                                "event: failed\ndata: {{\"reason\":\"{}\"}}\n\n",
                                reason.replace('"', "\\\""),
                            )
                        }
                        _ => String::new(), // Idle — close without event
                    }
                };
                if !training_status_json.is_empty() {
                    let _ = write_all(&mut writer, training_status_json.as_bytes());
                }
                return;
            }
        }
    }
}

/// Writes all bytes to the writer, returning `Err` on any I/O failure.
fn write_all<W: Write>(w: &mut W, data: &[u8]) -> std::io::Result<()> {
    w.write_all(data)?;
    w.flush()
}
