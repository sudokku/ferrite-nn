use std::io::Cursor;
use tiny_http::Response;

use crate::state::{SharedState, TrainingStatus};
use crate::render::{render_page, Page};

// ---------------------------------------------------------------------------
// GET /evaluate
// ---------------------------------------------------------------------------

pub fn handle_get(state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let st   = state.lock().unwrap();
    let mask = st.tab_unlock_mask();

    let history  = st.epoch_history.clone();
    let training = &st.training;

    // Final metrics
    let last = history.last();
    let (train_loss, val_loss, train_acc, val_acc) = last.map(|s| (
        format!("{:.6}", s.train_loss),
        s.val_loss.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "—".into()),
        s.train_accuracy.map(|v| format!("{:.2}%", v * 100.0)).unwrap_or_else(|| "—".into()),
        s.val_accuracy.map(|v| format!("{:.2}%", v * 100.0)).unwrap_or_else(|| "—".into()),
    )).unwrap_or_else(|| ("—".into(), "—".into(), "—".into(), "—".into()));

    let total_time = match training {
        TrainingStatus::Done { elapsed_total_ms, .. } =>
            format!("{:.1}s", *elapsed_total_ms as f64 / 1000.0),
        TrainingStatus::Stopped { epochs_completed } =>
            format!("stopped at {} epochs", epochs_completed),
        _ => "—".into(),
    };

    let epochs_ran = history.len();

    // SVG loss curve.
    let svg = build_svg_loss_curve(&history);

    // Metrics table.
    let metrics_table = format!(
        r#"<table class="summary-table">
          <tr><th>Epochs completed</th><td>{epochs}</td></tr>
          <tr><th>Final train loss</th><td>{train_loss}</td></tr>
          <tr><th>Final val loss</th><td>{val_loss}</td></tr>
          <tr><th>Train accuracy</th><td>{train_acc}</td></tr>
          <tr><th>Val accuracy</th><td>{val_acc}</td></tr>
          <tr><th>Total training time</th><td>{time}</td></tr>
        </table>"#,
        epochs = epochs_ran,
        train_loss = train_loss, val_loss = val_loss,
        train_acc = train_acc, val_acc = val_acc,
        time = total_time,
    );

    // Confusion matrix from trained network on validation set.
    let confusion_html = if let (Some(network_ref), Some(ds)) = (&st.trained_network, &st.dataset) {
        if !ds.val_inputs.is_empty() {
            let mut net = network_ref.clone();
            build_confusion_matrix_html(&mut net, &ds.val_inputs, &ds.val_labels)
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    drop(st);

    crate::routes::html_response(render_page(Page::Evaluate, mask, false, |tmpl| {
        tmpl
            .replace("{{EVAL_LOSS_SVG}}", &svg)
            .replace("{{EVAL_METRICS_TABLE}}", &metrics_table)
            .replace("{{EVAL_CONFUSION}}", &confusion_html)
    }))
}

// ---------------------------------------------------------------------------
// GET /evaluate/export
// ---------------------------------------------------------------------------

pub fn handle_export(state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let st      = state.lock().unwrap();
    let history = st.epoch_history.clone();
    drop(st);

    let json = serde_json::to_string_pretty(&history).unwrap_or_else(|_| "[]".into());
    crate::routes::json_download_response(json, "epoch_history.json")
}

// ---------------------------------------------------------------------------
// SVG loss curve
// ---------------------------------------------------------------------------

fn build_svg_loss_curve(history: &[ferrite_nn::EpochStats]) -> String {
    if history.len() < 2 {
        return "<p class=\"hint\">Not enough data to draw a curve.</p>".into();
    }

    let w = 760.0f64;
    let h = 220.0f64;
    let pad_l = 60.0f64;
    let pad_r = 16.0f64;
    let pad_t = 16.0f64;
    let pad_b = 30.0f64;

    let train_pts: Vec<f64> = history.iter().map(|s| s.train_loss).collect();
    let val_pts:   Vec<f64> = history.iter().filter_map(|s| s.val_loss).collect();

    let all_vals: Vec<f64> = train_pts.iter().chain(val_pts.iter()).cloned().collect();
    let max_y = all_vals.iter().cloned().fold(0.0f64, f64::max) * 1.05;
    let min_y = 0.0f64;
    let n     = train_pts.len();

    let px = |i: usize, v: f64| -> (f64, f64) {
        let x = pad_l + (i as f64 / (n - 1) as f64) * (w - pad_l - pad_r);
        let y = pad_t + (max_y - v) / (max_y - min_y + 1e-12) * (h - pad_t - pad_b);
        (x, y)
    };

    // Train polyline.
    let train_path: String = train_pts.iter().enumerate().map(|(i, &v)| {
        let (x, y) = px(i, v);
        if i == 0 { format!("M{:.1},{:.1}", x, y) } else { format!(" L{:.1},{:.1}", x, y) }
    }).collect();

    // Val polyline (if available, same count).
    let val_path: String = if val_pts.len() == n {
        val_pts.iter().enumerate().map(|(i, &v)| {
            let (x, y) = px(i, v);
            if i == 0 { format!("M{:.1},{:.1}", x, y) } else { format!(" L{:.1},{:.1}", x, y) }
        }).collect()
    } else {
        String::new()
    };

    // Y axis labels.
    let grey_grid = "#f0f2f5";
    let grey_text = "#999";
    let y_labels: String = (0..=4).map(|g| {
        let frac = g as f64 / 4.0;
        let val  = min_y + (max_y - min_y) * frac;
        let y    = pad_t + (1.0 - frac) * (h - pad_t - pad_b);
        let w_r  = w - pad_r;
        format!(
            "<text x=\"{}\" y=\"{:.1}\" text-anchor=\"end\" fill=\"{}\" font-size=\"10\">{:.3}</text>\n\
             <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"1\"/>",
            pad_l - 4.0, y + 4.0, grey_text, val,
            pad_l, y, w_r, y, grey_grid
        )
    }).collect::<Vec<_>>().join("\n");

    // X axis labels.
    let x_labels: String = [0, n / 2, n - 1].iter().map(|&i| {
        let (x, _) = px(i, 0.0);
        format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" fill=\"{}\" font-size=\"10\">{}</text>",
            x, h - 4.0, grey_text, i + 1
        )
    }).collect::<Vec<_>>().join("\n");

    let blue_dark  = "#1e40af";
    let red_dark   = "#dc2626";
    let dark_text  = "#333";

    let val_line = if !val_path.is_empty() {
        format!(
            "<path d=\"{}\" stroke=\"{}\" stroke-width=\"1.5\" fill=\"none\" stroke-dasharray=\"5,4\"/>",
            val_path, blue_dark
        )
    } else {
        String::new()
    };

    let val_legend = if !val_path.is_empty() {
        format!(
            "<line x1=\"120\" y1=\"9\" x2=\"138\" y2=\"9\" stroke=\"{}\" stroke-width=\"1.5\" stroke-dasharray=\"4,3\"/>\n\
             <text x=\"142\" y=\"13\" fill=\"{}\" font-size=\"10\">val loss</text>",
            blue_dark, dark_text
        )
    } else {
        String::new()
    };

    let ll = pad_l + 22.0;
    format!(
        "<svg class=\"loss-svg\" width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">\n\
         {}\n{}\n\
         <path d=\"{}\" stroke=\"{}\" stroke-width=\"2\" fill=\"none\"/>\n\
         {}\n\
         <!-- Legend -->\n\
         <rect x=\"{:.1}\" y=\"4\" width=\"18\" height=\"4\" fill=\"{}\"/>\n\
         <text x=\"{:.1}\" y=\"13\" fill=\"{}\" font-size=\"10\">train loss</text>\n\
         {}\n\
         </svg>",
        w, h,
        y_labels, x_labels,
        train_path, red_dark,
        val_line,
        pad_l, red_dark,
        ll, dark_text,
        val_legend,
    )
}

// ---------------------------------------------------------------------------
// Confusion matrix
// ---------------------------------------------------------------------------

fn build_confusion_matrix_html(
    network: &mut ferrite_nn::Network,
    val_inputs: &[Vec<f64>],
    val_labels: &[Vec<f64>],
) -> String {
    if val_labels.is_empty() { return String::new(); }

    let n_classes = val_labels[0].len();
    if n_classes < 2 { return String::new(); }

    let mut matrix = vec![vec![0usize; n_classes]; n_classes];

    for (input, label) in val_inputs.iter().zip(val_labels.iter()) {
        let output = network.forward(input.clone());
        let predicted = argmax(&output);
        let truth     = argmax(label);
        if predicted < n_classes && truth < n_classes {
            matrix[truth][predicted] += 1;
        }
    }

    let max_off_diag = matrix.iter().enumerate()
        .flat_map(|(r, row)| row.iter().enumerate().filter(move |(c, _)| *c != r).map(|(_, &v)| v))
        .max()
        .unwrap_or(1)
        .max(1);

    let header: String = (0..n_classes).map(|c| format!("<th>P:{}</th>", c)).collect();
    let rows: String = matrix.iter().enumerate().map(|(r, row)| {
        let cells: String = row.iter().enumerate().map(|(c, &v)| {
            if r == c {
                format!("<td class=\"conf-diag\">{}</td>", v)
            } else {
                let alpha = (v as f64 / max_off_diag as f64 * 0.4).min(0.4);
                let style = if v > 0 {
                    format!(" style=\"background:rgba(220,38,38,{:.2})\"", alpha)
                } else {
                    String::new()
                };
                format!("<td{}>{}</td>", style, v)
            }
        }).collect();
        format!("<tr><th>T:{}</th>{}</tr>", r, cells)
    }).collect();

    format!(
        r#"<div class="card"><h2>Confusion Matrix (Validation Set)</h2>
<p class="hint" style="margin-bottom:10px">Rows = true class, Columns = predicted class. Green diagonal = correct predictions.</p>
<div style="overflow-x:auto">
<table class="conf-matrix">
  <thead><tr><th></th>{header}</tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
</div>"#,
        header = header, rows = rows
    )
}

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
