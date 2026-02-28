use tiny_http::{Request, Response};
use std::io::Cursor;

use crate::state::{DatasetState, FlashMessage, SharedState};
use crate::util::form::{parse_form, form_get};
use crate::util::multipart::{extract_boundary, multipart_extract_file,
                              extract_all_text_fields};
use crate::util::csv::{parse_csv, LabelMode, builtin_xor, builtin_circles, builtin_blobs};
use crate::render::{render_page, Page};
use crate::handlers::architect::{render_flash_html, html_escape};

const MAX_CSV_BYTES: usize = 50 * 1024 * 1024; // 50 MB

// ---------------------------------------------------------------------------
// GET /dataset
// ---------------------------------------------------------------------------

pub fn handle_get(state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let mut st = state.lock().unwrap();
    let flash  = st.take_flash();
    let mask   = st.tab_unlock_mask();
    let ds     = st.dataset.clone();
    drop(st);

    crate::routes::html_response(build_dataset_page(&ds, None, flash, mask, "upload"))
}

// ---------------------------------------------------------------------------
// POST /dataset/upload
// ---------------------------------------------------------------------------

pub fn handle_upload(request: &mut Request, state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let content_type = request.headers().iter()
        .find(|h| h.field.equiv("Content-Type"))
        .map(|h| h.value.as_str().to_owned())
        .unwrap_or_default();

    let boundary = match extract_boundary(&content_type) {
        Some(b) => b,
        None    => return show_error(&state, "Invalid multipart request.", "upload"),
    };

    let mut body: Vec<u8> = Vec::new();
    let _ = request.as_reader().read_to_end(&mut body);

    if body.len() > MAX_CSV_BYTES {
        return show_error(&state, "File exceeds 50 MB limit.", "upload");
    }

    let csv_bytes = match multipart_extract_file(&body, &boundary) {
        Some(b) if !b.is_empty() => b,
        _ => return show_error(&state, "No CSV file was uploaded.", "upload"),
    };

    // Parse text fields from multipart.
    let fields = extract_all_text_fields(&body, &boundary);
    let field_get = |k: &str| fields.iter().find(|(name,_)| name == k).map(|(_,v)| v.as_str()).unwrap_or("");

    let val_split: u8 = field_get("val_split").trim().parse().unwrap_or(20).min(50);
    let label_mode_s  = field_get("label_mode");
    let n_classes: usize  = field_get("n_classes").trim().parse().unwrap_or(2).max(2);
    let n_label_cols: usize = field_get("n_label_cols").trim().parse().unwrap_or(1).max(1);

    let label_mode = if label_mode_s == "one_hot" {
        LabelMode::OneHot { n_label_cols }
    } else {
        LabelMode::ClassIndex { n_classes }
    };

    let (inputs, labels) = match parse_csv(&csv_bytes, label_mode) {
        Ok(r)  => r,
        Err(e) => return show_error(&state, &e.to_string(), "upload"),
    };

    // Validate feature count against spec.
    {
        let st = state.lock().unwrap();
        if let Some(spec) = &st.spec {
            let expected = spec.layers.first().map(|l| l.input_size).unwrap_or(0);
            if expected > 0 && inputs[0].len() != expected {
                let err = format!(
                    "Feature count mismatch: model expects {} inputs, CSV has {}.",
                    expected, inputs[0].len()
                );
                drop(st);
                return show_error(&state, &err, "upload");
            }
        }
    }

    let ds = build_dataset_state(inputs, labels, val_split, "CSV upload".to_owned());

    let mut st = state.lock().unwrap();
    st.dataset = Some(ds);
    st.flash   = Some(FlashMessage::success("Dataset loaded successfully."));
    drop(st);

    crate::routes::redirect("/dataset")
}

// ---------------------------------------------------------------------------
// POST /dataset/builtin
// ---------------------------------------------------------------------------

pub fn handle_builtin(request: &mut Request, state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let mut body = String::new();
    let _ = request.as_reader().read_to_string(&mut body);
    let pairs = parse_form(&body);

    let name      = form_get(&pairs, "builtin_name").unwrap_or("xor");
    // XOR has only 4 samples — any validation split causes misleading metrics
    // because the model never sees the held-out sample(s) during training.
    let val_split: u8 = if name == "xor" {
        0
    } else {
        form_get(&pairs, "val_split")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(20)
            .min(50)
    };

    let (inputs, labels, source_name) = match name {
        "circles" => { let (i,l) = builtin_circles(200); (i, l, "Circles (200)".to_owned()) }
        "blobs"   => { let (i,l) = builtin_blobs(200);   (i, l, "Blobs (200)".to_owned())   }
        "mnist"   => {
            // MNIST is only available if IDX files exist.
            return show_error(&state, "MNIST dataset not implemented in built-in loader; train with examples/mnist.rs first.", "builtin");
        }
        _         => { let (i,l) = builtin_xor();        (i, l, "XOR".to_owned())            }
    };

    // Validate feature count.
    {
        let st = state.lock().unwrap();
        if let Some(spec) = &st.spec {
            let expected = spec.layers.first().map(|l| l.input_size).unwrap_or(0);
            if expected > 0 && !inputs.is_empty() && inputs[0].len() != expected {
                let err = format!(
                    "Feature count mismatch: model expects {} inputs, '{}' has {}.",
                    expected, name, inputs[0].len()
                );
                drop(st);
                return show_error(&state, &err, "builtin");
            }
        }
    }

    let ds = build_dataset_state(inputs, labels, val_split, source_name);

    let mut st = state.lock().unwrap();
    st.dataset = Some(ds);
    st.flash   = Some(FlashMessage::success("Dataset loaded successfully."));
    drop(st);

    crate::routes::redirect("/dataset")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn show_error(state: &SharedState, msg: &str, active_panel: &str) -> Response<Cursor<Vec<u8>>> {
    let st   = state.lock().unwrap();
    let mask = st.tab_unlock_mask();
    let ds   = st.dataset.clone();
    drop(st);
    crate::routes::html_response(build_dataset_page(&ds, Some(msg), None, mask, active_panel))
}

fn build_dataset_state(
    inputs: Vec<Vec<f64>>,
    labels: Vec<Vec<f64>>,
    val_split_pct: u8,
    source_name: String,
) -> DatasetState {
    let total = inputs.len();
    let feature_count = inputs.first().map(|r| r.len()).unwrap_or(0);
    let label_count   = labels.first().map(|r| r.len()).unwrap_or(0);

    let val_n = (total * val_split_pct as usize) / 100;
    let train_n = total - val_n;

    let preview_rows: Vec<(Vec<f64>, Vec<f64>)> = inputs.iter().zip(labels.iter())
        .take(5)
        .map(|(i, l)| (i.clone(), l.clone()))
        .collect();

    let (train_inputs, val_inputs) = inputs.split_at(train_n);
    let (train_labels, val_labels) = labels.split_at(train_n);

    DatasetState {
        train_inputs:  train_inputs.to_vec(),
        train_labels:  train_labels.to_vec(),
        val_inputs:    val_inputs.to_vec(),
        val_labels:    val_labels.to_vec(),
        feature_count,
        label_count,
        total_rows: total,
        val_split_pct,
        source_name,
        preview_rows,
    }
}

fn build_dataset_page(
    ds:           &Option<DatasetState>,
    error:        Option<&str>,
    flash:        Option<FlashMessage>,
    tab_unlock:   u8,
    active_panel: &str,
) -> String {
    let flash_html = render_flash_html(flash.as_ref());
    let error_html = error.map(|e| {
        format!(r#"<div class="flash flash-error" style="margin-top:14px">{}</div>"#, html_escape(e))
    }).unwrap_or_default();

    let upload_active  = if active_panel == "upload" { "active" } else { "" };
    let builtin_active = if active_panel == "builtin" { "active" } else { "" };
    let upload_hide    = if active_panel == "builtin" { "hidden" } else { "" };
    let builtin_hide   = if active_panel == "upload" { "hidden" } else { "" };

    let summary_html = ds.as_ref().map(build_summary_html).unwrap_or_default();

    render_page(Page::Dataset, tab_unlock, false, |tmpl| {
        tmpl
            .replace("{{FLASH_DATASET}}", &flash_html)
            .replace("{{DS_UPLOAD_ACTIVE}}", upload_active)
            .replace("{{DS_BUILTIN_ACTIVE}}", builtin_active)
            .replace("{{DS_UPLOAD_HIDE}}", upload_hide)
            .replace("{{DS_BUILTIN_HIDE}}", builtin_hide)
            .replace("{{DS_VAL_SPLIT}}", "20")
            .replace("{{SEL_CI}}", " selected")
            .replace("{{SEL_OH}}", "")
            .replace("{{N_CLASSES_HIDE}}", "")
            .replace("{{N_LABEL_COLS_HIDE}}", "hidden")
            .replace("{{DS_N_CLASSES}}", "2")
            .replace("{{DS_N_LABEL_COLS}}", "1")
            .replace("{{SEL_XOR}}", "checked")
            .replace("{{SEL_CIRCLES}}", "")
            .replace("{{SEL_BLOBS}}", "")
            .replace("{{MNIST_OPTION}}", "")
            .replace("{{DS_ERROR}}", &error_html)
            .replace("{{DS_SUMMARY}}", &summary_html)
    })
}

fn build_summary_html(ds: &DatasetState) -> String {
    let preview: String = ds.preview_rows.iter().enumerate().map(|(i, (inp, lbl))| {
        let feat_str: String = inp.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(", ");
        let lbl_str:  String = lbl.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(", ");
        format!("<tr><td>{}</td><td>{}</td><td>{}</td></tr>", i+1, html_escape(&feat_str), html_escape(&lbl_str))
    }).collect();

    format!(
        r#"<div class="card"><h2>Dataset Summary</h2>
<table class="summary-table">
  <tr><th>Source</th><td>{source}</td></tr>
  <tr><th>Total rows</th><td>{total}</td></tr>
  <tr><th>Features</th><td>{feats}</td></tr>
  <tr><th>Labels</th><td>{lbls}</td></tr>
  <tr><th>Training samples</th><td>{train_n}</td></tr>
  <tr><th>Validation samples</th><td>{val_n}</td></tr>
  <tr><th>Validation split</th><td>{split}%</td></tr>
</table>
<h3 style="margin-top:18px">First {preview_count} rows</h3>
<table class="preview-table">
  <thead><tr><th>#</th><th>Features</th><th>Labels</th></tr></thead>
  <tbody>{preview}</tbody>
</table>
</div>"#,
        source       = html_escape(&ds.source_name),
        total        = ds.total_rows,
        feats        = ds.feature_count,
        lbls         = ds.label_count,
        train_n      = ds.train_inputs.len(),
        val_n        = ds.val_inputs.len(),
        split        = ds.val_split_pct,
        preview_count = ds.preview_rows.len(),
        preview      = preview,
    )
}
