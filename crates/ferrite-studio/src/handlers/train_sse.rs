use std::convert::Infallible;
use std::time::Duration;

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
};
use futures::stream::{self, Stream, StreamExt};

use crate::routes::SharedState;
use crate::state::TrainingStatus;

/// Type alias to avoid repeating the boxed stream signature everywhere.
type SseStream = std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

/// `GET /api/train/events` — Server-Sent Events endpoint.
///
/// Streams epoch stats as SSE events. The event format is preserved exactly:
///
/// ```
/// event: epoch
/// data: {"epoch":1,"total_epochs":50,"train_loss":0.5,...}
///
/// event: done
/// data: {"model_path":"trained_models/my_model.json","elapsed_total_ms":12345,"epochs_completed":50}
///
/// event: stopped
/// data: {"model_path":"...","elapsed_total_ms":3000,"epoch_reached":10,"total_epochs":50}
///
/// event: failed
/// data: {"reason":"..."}
/// ```
pub async fn handle(State(state): State<SharedState>) -> Sse<SseStream> {
    // Clone the epoch receiver arc out of state before building the stream,
    // so we do not hold the lock during the async stream.
    let epoch_rx = {
        let st = state.lock().unwrap();
        match &st.training {
            TrainingStatus::Running { epoch_rx, .. } => Some(epoch_rx.clone()),
            _ => None,
        }
    };

    // Replay any epoch stats already accumulated (handles SSE client reconnects).
    let history_events: Vec<Result<Event, Infallible>> = {
        let st = state.lock().unwrap();
        st.epoch_history.iter()
            .filter_map(|stats| {
                serde_json::to_string(stats).ok().map(|json| {
                    Ok(Event::default().event("epoch").data(json))
                })
            })
            .collect()
    };

    let stream: SseStream = match epoch_rx {
        None => {
            // Not currently training — emit the terminal event for the current status.
            let terminal_event = {
                let st = state.lock().unwrap();
                match &st.training {
                    TrainingStatus::Done { model_path, elapsed_total_ms, was_stopped } => {
                        let ep    = st.epoch_history.len();
                        let total = st.hyperparams.as_ref().map(|h| h.epochs).unwrap_or(0);
                        let (event_name, json) = if *was_stopped {
                            (
                                "stopped",
                                format!(
                                    "{{\"model_path\":\"{mp}\",\"elapsed_total_ms\":{el},\"epoch_reached\":{ep},\"total_epochs\":{total}}}",
                                    mp = model_path, el = elapsed_total_ms, ep = ep, total = total,
                                ),
                            )
                        } else {
                            (
                                "done",
                                format!(
                                    "{{\"model_path\":\"{mp}\",\"elapsed_total_ms\":{el},\"epochs_completed\":{ep}}}",
                                    mp = model_path, el = elapsed_total_ms, ep = ep,
                                ),
                            )
                        };
                        Some(Ok(Event::default().event(event_name).data(json)))
                    }
                    TrainingStatus::Failed { reason } => {
                        let json = format!(
                            "{{\"reason\":\"{}\"}}",
                            reason.replace('"', "\\\"")
                        );
                        Some(Ok(Event::default().event("failed").data(json)))
                    }
                    _ => None, // Idle — close without event
                }
            };

            let events: Vec<Result<Event, Infallible>> = history_events
                .into_iter()
                .chain(terminal_event.into_iter())
                .collect();

            Box::pin(stream::iter(events))
        }
        Some(rx_arc) => {
            // Training is running — stream epoch events from the tokio mpsc channel.
            let state_for_stream = state.clone();

            let live_stream = stream::unfold(
                (rx_arc, state_for_stream, false),
                |(rx_arc, state, done)| async move {
                    if done {
                        return None;
                    }

                    // Try to receive the next epoch stats with a 500 ms timeout.
                    let recv_result = {
                        let mut rx = rx_arc.lock().await;
                        tokio::time::timeout(Duration::from_millis(500), rx.recv()).await
                    };

                    match recv_result {
                        Ok(Some(stats)) => {
                            // Accumulate stats in epoch_history.
                            {
                                let mut st = state.lock().unwrap();
                                st.epoch_history.push(stats.clone());
                            }
                            let event = match serde_json::to_string(&stats) {
                                Ok(json) => Ok(Event::default().event("epoch").data(json)),
                                Err(_)   => Ok(Event::default().event("epoch").data("{}")),
                            };
                            Some((event, (rx_arc, state, false)))
                        }
                        Ok(None) => {
                            // Channel closed — training finished. Emit terminal event.
                            let terminal = build_terminal_event(&state);
                            // Signal done so the stream ends after this event.
                            Some((terminal, (rx_arc, state, true)))
                        }
                        Err(_timeout) => {
                            // Timeout — emit an SSE comment (keepalive).
                            Some((
                                Ok(Event::default().comment("ping")),
                                (rx_arc, state, false),
                            ))
                        }
                    }
                },
            );

            let combined = stream::iter(history_events).chain(live_stream);
            Box::pin(combined)
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

/// Builds the terminal SSE event (done / stopped / failed) from current state.
fn build_terminal_event(state: &SharedState) -> Result<Event, Infallible> {
    let st = state.lock().unwrap();
    match &st.training {
        TrainingStatus::Done { model_path, elapsed_total_ms, was_stopped } => {
            let ep    = st.epoch_history.len();
            let total = st.hyperparams.as_ref().map(|h| h.epochs).unwrap_or(0);
            let (event_name, json) = if *was_stopped {
                (
                    "stopped",
                    format!(
                        "{{\"model_path\":\"{mp}\",\"elapsed_total_ms\":{el},\"epoch_reached\":{ep},\"total_epochs\":{total}}}",
                        mp = model_path, el = elapsed_total_ms, ep = ep, total = total,
                    ),
                )
            } else {
                (
                    "done",
                    format!(
                        "{{\"model_path\":\"{mp}\",\"elapsed_total_ms\":{el},\"epochs_completed\":{ep}}}",
                        mp = model_path, el = elapsed_total_ms, ep = ep,
                    ),
                )
            };
            Ok(Event::default().event(event_name).data(json))
        }
        TrainingStatus::Failed { reason } => {
            let json = format!("{{\"reason\":\"{}\"}}", reason.replace('"', "\\\""));
            Ok(Event::default().event("failed").data(json))
        }
        _ => {
            // Idle or unexpected state.
            Ok(Event::default().event("done").data("{}"))
        }
    }
}
