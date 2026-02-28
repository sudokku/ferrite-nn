use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}, mpsc};
use std::thread;
use tiny_http::Response;
use std::io::Cursor;

use ferrite_nn::{Network, Sgd, LossType, TrainConfig, train_loop};

use crate::state::{FlashMessage, SharedState, TrainingStatus};
use crate::render::{render_page, Page};
use crate::handlers::architect::{render_flash_html, html_escape, activation_to_str};

// ---------------------------------------------------------------------------
// GET /train
// ---------------------------------------------------------------------------

pub fn handle_get(state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let mut st = state.lock().unwrap();
    let flash      = st.take_flash();
    let mask       = st.tab_unlock_mask();
    let spec       = st.spec.clone();
    let hp         = st.hyperparams.clone();
    let ds         = st.dataset.as_ref().map(|d| (d.train_inputs.len(), d.val_inputs.len(), d.source_name.clone()));
    let training   = &st.training;
    let history    = st.epoch_history.clone();

    let (show_summary, show_live, show_done, show_failed) = match training {
        TrainingStatus::Idle           => (true,  false, false, false),
        TrainingStatus::Running { .. } => (false, true,  false, false),
        TrainingStatus::Done    { .. } => (false, false, true,  false),
        TrainingStatus::Failed  { .. } => (false, false, false, true),
    };

    let is_running = matches!(training, TrainingStatus::Running { .. });

    let total_epochs = match training {
        TrainingStatus::Running { total_epochs, .. } => *total_epochs,
        _ => hp.as_ref().map(|h| h.epochs).unwrap_or(50),
    };

    let done_badge = match training {
        TrainingStatus::Done { was_stopped: true,  .. } => "Stopped",
        TrainingStatus::Done { was_stopped: false, .. } => "Done",
        _ => "",
    };

    let done_stats_html = build_done_stats(&st.training, &history);
    let download_link   = build_download_link(&st.training);
    let fail_reason     = match &st.training {
        TrainingStatus::Failed { reason } => reason.clone(),
        _ => String::new(),
    };
    let train_error = if spec.is_none() || ds.is_none() {
        "<div class=\"flash flash-error\">Set up architecture and dataset first.</div>"
    } else {
        ""
    };

    drop(st);

    let arch_summary = spec.as_ref().map(|s| {
        let layers_desc: String = s.layers.iter().enumerate().map(|(i, l)| {
            format!("<div class=\"arch-row\"><span class=\"ar-lbl\">Layer {}</span><span class=\"ar-val\">{} neurons — {}</span></div>",
                i+1, l.size, activation_to_str(&l.activation))
        }).collect();
        let loss_name = if s.loss == LossType::CrossEntropy { "Cross-Entropy" } else { "MSE" };
        format!(
            r#"<div class="arch-summary-grid" style="margin-bottom:12px">
              <div class="arch-row"><span class="ar-lbl">Model name</span><span class="ar-val">{name}</span></div>
              <div class="arch-row"><span class="ar-lbl">Input size</span><span class="ar-val">{input_size}</span></div>
              {layers}
              <div class="arch-row"><span class="ar-lbl">Loss</span><span class="ar-val">{loss}</span></div>
            </div>"#,
            name       = html_escape(&s.name),
            input_size = s.layers.first().map(|l| l.input_size).unwrap_or(0),
            layers     = layers_desc,
            loss       = loss_name,
        )
    }).unwrap_or_else(|| "<p class=\"hint\">No architecture saved yet.</p>".into());

    let data_summary = ds.map(|(train_n, val_n, src)| {
        format!(
            r#"<div class="arch-summary-grid"><div class="arch-row"><span class="ar-lbl">Dataset</span><span class="ar-val">{src}</span></div><div class="arch-row"><span class="ar-lbl">Train samples</span><span class="ar-val">{train_n}</span></div><div class="arch-row"><span class="ar-lbl">Val samples</span><span class="ar-val">{val_n}</span></div></div>"#,
            src = html_escape(&src), train_n = train_n, val_n = val_n
        )
    }).unwrap_or_else(|| "<p class=\"hint\">No dataset loaded yet.</p>".into());

    let flash_html = render_flash_html(flash.as_ref());

    let hide  = |show: bool| if show { "" } else { "hidden" };

    crate::routes::html_response(render_page(Page::Train, mask, is_running, |tmpl| {
        tmpl
            .replace("{{FLASH_TRAIN}}", &flash_html)
            .replace("{{TRAIN_SUMMARY_HIDE}}", hide(show_summary))
            .replace("{{TRAIN_LIVE_HIDE}}", hide(show_live))
            .replace("{{TRAIN_DONE_HIDE}}", hide(show_done))
            .replace("{{TRAIN_FAILED_HIDE}}", hide(show_failed))
            .replace("{{TRAIN_ARCH_SUMMARY}}", &arch_summary)
            .replace("{{TRAIN_DATA_SUMMARY}}", &data_summary)
            .replace("{{TRAIN_TOTAL_EPOCHS}}", &total_epochs.to_string())
            .replace("{{TRAIN_STATUS_BADGE}}", done_badge)
            .replace("{{TRAIN_DONE_STATS}}", &done_stats_html)
            .replace("{{TRAIN_DOWNLOAD_LINK}}", &download_link)
            .replace("{{TRAIN_FAIL_REASON}}", &html_escape(&fail_reason))
            .replace("{{TRAIN_ERROR}}", train_error)
    }))
}

fn build_done_stats(training: &TrainingStatus, history: &[ferrite_nn::EpochStats]) -> String {
    let last = history.last();
    let (train_loss, val_loss, train_acc, val_acc) = last.map(|s| (
        format!("{:.6}", s.train_loss),
        s.val_loss.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "—".into()),
        s.train_accuracy.map(|v| format!("{:.2}%", v * 100.0)).unwrap_or_else(|| "—".into()),
        s.val_accuracy.map(|v| format!("{:.2}%", v * 100.0)).unwrap_or_else(|| "—".into()),
    )).unwrap_or_else(|| ("—".into(), "—".into(), "—".into(), "—".into()));

    let (elapsed_total, saved_path) = match training {
        TrainingStatus::Done { elapsed_total_ms, model_path, was_stopped } => {
            let elapsed = if *was_stopped {
                format!("stopped at epoch {}", history.len())
            } else {
                format!("{:.1}s", *elapsed_total_ms as f64 / 1000.0)
            };
            (elapsed, model_path.clone())
        }
        _ => ("—".into(), String::new()),
    };

    let saved_line = if saved_path.is_empty() {
        String::new()
    } else {
        format!(
            r#"<p style="margin-top:12px;font-size:.85rem;color:#555">Saved to: <code>{}</code></p>"#,
            html_escape(&saved_path)
        )
    };

    format!(
        r#"<div class="metrics-row" id="done-stats-js">
          <div class="metric-card"><div class="val">{train_loss}</div><div class="lbl">Train loss</div></div>
          <div class="metric-card"><div class="val">{val_loss}</div><div class="lbl">Val loss</div></div>
          <div class="metric-card"><div class="val">{train_acc}</div><div class="lbl">Train acc</div></div>
          <div class="metric-card"><div class="val">{val_acc}</div><div class="lbl">Val acc</div></div>
          <div class="metric-card"><div class="val" style="font-size:1rem">{elapsed}</div><div class="lbl">Total time</div></div>
        </div>
        {saved_line}
        <div id="done-download-js"></div>"#,
        train_loss  = train_loss,
        val_loss    = val_loss,
        train_acc   = train_acc,
        val_acc     = val_acc,
        elapsed     = elapsed_total,
        saved_line  = saved_line,
    )
}

fn build_download_link(training: &TrainingStatus) -> String {
    match training {
        TrainingStatus::Done { model_path, .. } => {
            // Extract stem from path for the download route.
            let stem = std::path::Path::new(model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            format!(
                r#"<a href="/models/{stem}/download" class="btn btn-secondary">Download model JSON</a>"#,
                stem = html_escape(stem)
            )
        }
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// POST /train/start
// ---------------------------------------------------------------------------

pub fn handle_start(state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let mut st = state.lock().unwrap();

    // Guard: need spec + hyperparams + dataset.
    if st.spec.is_none() || st.hyperparams.is_none() || st.dataset.is_none() {
        st.flash = Some(FlashMessage::error("Set up architecture and dataset before training."));
        drop(st);
        return crate::routes::redirect("/train");
    }

    // If already running, don't start another.
    if matches!(st.training, TrainingStatus::Running { .. }) {
        drop(st);
        return crate::routes::redirect("/train");
    }

    let spec   = st.spec.clone().unwrap();
    let hp     = st.hyperparams.clone().unwrap();
    let ds     = st.dataset.clone().unwrap();

    let (tx, rx) = mpsc::channel::<ferrite_nn::EpochStats>();
    let stop_flag = Arc::new(AtomicBool::new(false));

    let epoch_rx = Arc::new(Mutex::new(rx));
    let total_epochs = hp.epochs;

    st.training = TrainingStatus::Running {
        stop_flag:   stop_flag.clone(),
        epoch_rx:    epoch_rx.clone(),
        total_epochs,
    };
    st.epoch_history.clear();
    st.trained_network = None;
    drop(st);

    // Spawn background training thread.
    let state_clone = state.clone();
    thread::spawn(move || {
        let mut network = Network::from_spec(&spec);
        let optimizer   = Sgd::new(hp.learning_rate);

        let val_inputs = if ds.val_inputs.is_empty() { None } else { Some(ds.val_inputs.as_slice()) };
        let val_labels = if ds.val_labels.is_empty() { None } else { Some(ds.val_labels.as_slice()) };

        let mut config = TrainConfig::new(hp.epochs, hp.batch_size, spec.loss);
        config.progress_tx = Some(tx);
        config.stop_flag   = Some(stop_flag.clone());

        let t_start = std::time::Instant::now();

        train_loop(
            &mut network,
            &ds.train_inputs,
            &ds.train_labels,
            val_inputs,
            val_labels,
            &optimizer,
            &config,
        );

        let elapsed_total_ms = t_start.elapsed().as_millis() as u64;
        let was_stopped = stop_flag.load(Ordering::Relaxed);

        // Save model.
        let model_name = spec.name.clone();
        let model_dir  = "trained_models";
        let model_path = format!("{}/{}.json", model_dir, model_name);
        let _ = std::fs::create_dir_all(model_dir);
        // Attach metadata from spec.
        network.metadata = spec.metadata.clone();
        let save_ok = network.save_json(&model_path).is_ok();

        let mut st = state_clone.lock().unwrap();

        // Drain any remaining EpochStats from the channel into a local buffer
        // first, then push them — avoids holding an immutable borrow on
        // `st.training` while mutably borrowing `st.epoch_history`.
        let remaining: Vec<ferrite_nn::EpochStats> = {
            if let TrainingStatus::Running { epoch_rx, .. } = &st.training {
                let rx_guard = epoch_rx.lock().unwrap();
                let mut buf = Vec::new();
                while let Ok(s) = rx_guard.try_recv() {
                    buf.push(s);
                }
                buf
            } else {
                Vec::new()
            }
        };
        for s in remaining {
            st.epoch_history.push(s);
        }

        if save_ok {
            // Model saved — always transition to Done, regardless of whether
            // the user clicked Stop. `was_stopped` lets the UI distinguish.
            st.training = TrainingStatus::Done {
                model_path: model_path.clone(),
                elapsed_total_ms,
                was_stopped,
            };
        } else {
            st.training = TrainingStatus::Failed {
                reason: format!("Training finished but could not save model to '{}'.", model_path),
            };
        }
        st.trained_network = Some(network);
    });

    crate::routes::redirect("/train")
}

// ---------------------------------------------------------------------------
// POST /train/stop
// ---------------------------------------------------------------------------

pub fn handle_stop(state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let st = state.lock().unwrap();
    if let TrainingStatus::Running { stop_flag, .. } = &st.training {
        stop_flag.store(true, Ordering::Relaxed);
    }
    drop(st);
    crate::routes::redirect("/train")
}
