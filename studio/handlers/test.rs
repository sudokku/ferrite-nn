use std::io::Cursor;
use tiny_http::{Request, Response};

use ferrite_nn::{ActivationFunction, InputType, Network};

use crate::state::SharedState;
use crate::util::form::{parse_form, form_get};
use crate::util::multipart::{extract_boundary, multipart_extract_file, extract_text_field,
                              find_subsequence, split_on};
use crate::util::image::{image_bytes_to_grayscale_input, image_bytes_to_rgb_input};
use crate::render::{render_page, Page};
use crate::handlers::architect::html_escape;

// ---------------------------------------------------------------------------
// GET /test  and  GET /test?model=NAME
// ---------------------------------------------------------------------------

pub fn handle_get(query: String, state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let st   = state.lock().unwrap();
    let mask = st.tab_unlock_mask();
    drop(st);

    let q_pairs  = parse_form(&query);
    let selected = form_get(&q_pairs, "model").unwrap_or("").to_owned();

    let page = build_test_page(&selected, "", mask);
    crate::routes::html_response(page)
}

// ---------------------------------------------------------------------------
// POST /test/infer
// ---------------------------------------------------------------------------

pub fn handle_infer(request: &mut Request, state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let st   = state.lock().unwrap();
    let mask = st.tab_unlock_mask();
    drop(st);

    let content_type = request.headers().iter()
        .find(|h| h.field.equiv("Content-Type"))
        .map(|h| h.value.as_str().to_owned())
        .unwrap_or_default();

    let is_multipart = content_type.starts_with("multipart/form-data");

    let (model_name, result_html) = if is_multipart {
        let mut body_bytes: Vec<u8> = Vec::new();
        let _ = request.as_reader().read_to_end(&mut body_bytes);
        let boundary = extract_boundary(&content_type).unwrap_or_default();

        let model_name = extract_text_field(&body_bytes, &boundary, "model")
            .unwrap_or_default();

        let result = match multipart_extract_file(&body_bytes, &boundary) {
            Some(bytes) if !bytes.is_empty() => run_inference_image(&model_name, &bytes),
            _ => error_html("No image file was uploaded."),
        };
        (model_name, result)
    } else {
        let mut body = String::new();
        let _ = request.as_reader().read_to_string(&mut body);
        let pairs      = parse_form(&body);
        let model_name = form_get(&pairs, "model").unwrap_or("").to_owned();
        let raw_inputs = form_get(&pairs, "inputs").unwrap_or("").to_owned();
        let result     = run_inference_numeric(&model_name, &raw_inputs);
        (model_name, result)
    };

    let page = build_test_page(&model_name, &result_html, mask);
    crate::routes::html_response(page)
}

// ---------------------------------------------------------------------------
// Page builder
// ---------------------------------------------------------------------------

fn build_test_page(selected: &str, result_html: &str, tab_unlock: u8) -> String {
    let models = list_models();
    let model_options = build_model_options(&models, selected);
    let (form_enctype, input_section) = build_input_section(selected);

    let full_input_section = format!(
        r#"<form method="POST" action="/test/infer" enctype="{enctype}" style="margin-top:18px">
  <input type="hidden" name="model" value="{model}">
  {input}
  <div class="mt"><button type="submit" class="btn btn-primary">Run Inference</button></div>
</form>"#,
        enctype = form_enctype,
        model   = html_escape(selected),
        input   = input_section,
    );

    render_page(Page::Test, tab_unlock, false, |tmpl| {
        tmpl
            .replace("{{MODEL_OPTIONS}}", &model_options)
            .replace("{{TEST_INPUT_SECTION}}", &full_input_section)
            .replace("{{TEST_RESULT_SECTION}}", result_html)
    })
}

// ---------------------------------------------------------------------------
// Model listing
// ---------------------------------------------------------------------------

fn list_models() -> Vec<String> {
    let dir = "trained_models";
    match std::fs::read_dir(dir) {
        Ok(entries) => {
            let mut names: Vec<String> = entries.flatten()
                .filter_map(|e| {
                    let path = e.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json") {
                        path.file_stem().and_then(|s| s.to_str()).map(|s| s.to_owned())
                    } else {
                        None
                    }
                })
                .collect();
            names.sort();
            names
        }
        Err(_) => vec![],
    }
}

fn build_model_options(models: &[String], selected: &str) -> String {
    if models.is_empty() {
        return r#"<option disabled>No models found in trained_models/</option>"#.into();
    }
    models.iter().map(|name| {
        let sel = if name == selected { " selected" } else { "" };
        format!("<option value=\"{}\"{}>{}</option>", html_escape(name), sel, html_escape(name))
    }).collect::<Vec<_>>().join("\n")
}

// ---------------------------------------------------------------------------
// Input section (based on model metadata)
// ---------------------------------------------------------------------------

fn build_input_section(model_name: &str) -> (&'static str, String) {
    if model_name.is_empty() {
        return numeric_section();
    }
    let path = format!("trained_models/{}.json", model_name);
    let network = Network::load_json(&path).ok();
    let input_type = network.as_ref()
        .and_then(|n| n.metadata.as_ref())
        .and_then(|m| m.input_type.as_ref());

    match input_type {
        Some(InputType::ImageGrayscale { width, height }) => {
            image_section(*width, *height, "Grayscale")
        }
        Some(InputType::ImageRgb { width, height }) => {
            image_section(*width, *height, "RGB")
        }
        _ => numeric_section(),
    }
}

fn image_section(width: u32, height: u32, color_mode: &str) -> (&'static str, String) {
    let hint = format!("{} image — will be resized to {}x{} and normalized.", color_mode, width, height);
    (
        "multipart/form-data",
        format!(
            r#"<label for="image_file">Upload image</label>
<input type="file" id="image_file" name="image_file" accept="image/png,image/jpeg,image/bmp,image/gif" style="margin-bottom:10px">
<div id="preview-wrap" style="display:none;margin-bottom:10px">
  <img id="preview" style="max-width:140px;image-rendering:pixelated;border-radius:6px;border:1.5px solid #dde2ec">
</div>
<p class="hint">{hint}</p>
<script>
document.getElementById('image_file').addEventListener('change', function() {{
  var img = document.getElementById('preview');
  img.src = URL.createObjectURL(this.files[0]);
  document.getElementById('preview-wrap').style.display = 'block';
}});
</script>"#,
            hint = hint
        ),
    )
}

fn numeric_section() -> (&'static str, String) {
    (
        "application/x-www-form-urlencoded",
        r#"<label for="inputs">Input values</label>
<textarea id="inputs" name="inputs" rows="4"
  placeholder="Enter comma-separated numbers, e.g.:&#10;0.0, 1.0"></textarea>
<p class="hint">Comma-separated floats — one value per input neuron.</p>"#.to_owned(),
    )
}

// ---------------------------------------------------------------------------
// Inference runners
// ---------------------------------------------------------------------------

fn run_inference_numeric(model_name: &str, raw_inputs: &str) -> String {
    let path = format!("trained_models/{}.json", model_name);
    let mut network = match Network::load_json(&path) {
        Ok(n)  => n,
        Err(e) => return error_html(&format!("Could not load model <strong>{}</strong>: {}", html_escape(model_name), e)),
    };
    if network.layers.is_empty() { return error_html("Model has no layers."); }

    let inputs: Vec<f64> = raw_inputs
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();

    let expected_len = network.layers[0].weights.cols;
    if inputs.len() != expected_len {
        return error_html(&format!(
            "Input length mismatch: model expects <strong>{}</strong> values, got <strong>{}</strong>.",
            expected_len, inputs.len()
        ));
    }

    let output = network.forward(inputs);
    let labels = network.metadata.as_ref().and_then(|m| m.output_labels.as_deref());
    format_output(&output, labels, &network.layers.last().unwrap().activator)
}

fn run_inference_image(model_name: &str, image_bytes: &[u8]) -> String {
    let path = format!("trained_models/{}.json", model_name);
    let mut network = match Network::load_json(&path) {
        Ok(n)  => n,
        Err(e) => return error_html(&format!("Could not load model <strong>{}</strong>: {}", html_escape(model_name), e)),
    };
    if network.layers.is_empty() { return error_html("Model has no layers."); }

    let input_type = network.metadata.as_ref().and_then(|m| m.input_type.as_ref()).cloned();

    let inputs = match &input_type {
        Some(InputType::ImageGrayscale { width, height }) => {
            match image_bytes_to_grayscale_input(image_bytes, *width, *height) {
                Ok(v)  => v,
                Err(e) => return error_html(&format!("Image decode error: {}", e)),
            }
        }
        Some(InputType::ImageRgb { width, height }) => {
            match image_bytes_to_rgb_input(image_bytes, *width, *height) {
                Ok(v)  => v,
                Err(e) => return error_html(&format!("Image decode error: {}", e)),
            }
        }
        _ => return error_html("Model does not declare an image input type."),
    };

    let output = network.forward(inputs);
    let labels = network.metadata.as_ref().and_then(|m| m.output_labels.as_deref());
    format_output(&output, labels, &network.layers.last().unwrap().activator)
}

// ---------------------------------------------------------------------------
// Output formatters
// ---------------------------------------------------------------------------

fn format_output(output: &[f64], labels: Option<&[String]>, activator: &ActivationFunction) -> String {
    match activator {
        ActivationFunction::Softmax                         => format_softmax(output, labels),
        ActivationFunction::Sigmoid if output.len() == 1   => format_sigmoid(output[0]),
        _                                                   => format_raw(output),
    }
}

fn format_softmax(output: &[f64], labels: Option<&[String]>) -> String {
    let n = output.len();
    let (best, best_conf) = output.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));

    let label_for = |i: usize| -> String {
        labels.and_then(|l| l.get(i)).cloned().unwrap_or_else(|| i.to_string())
    };

    let hero = label_for(best);

    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| output[b].partial_cmp(&output[a]).unwrap());

    let rows: String = sorted.iter().map(|&i| {
        let pct   = output[i] * 100.0;
        let width = (output[i] * 260.0) as u32;
        let dim   = if i != best { " dim" } else { "" };
        format!(
            r#"<tr><td style="width:60px;font-weight:600;color:#333">{}</td><td><div class="bar-wrap"><div class="bar-fill{}" style="width:{}px"></div></div></td><td class="prob-pct">{:.1}%</td></tr>"#,
            label_for(i), dim, width, pct
        )
    }).collect();

    format!(
        r#"<div class="result-card"><h2>Result</h2>
<div class="prediction-hero">{hero}</div>
<div class="prediction-sub">Confidence: {conf:.1}%</div>
<table class="prob-table">
  <thead><tr><th>Class</th><th>Confidence</th><th></th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>"#,
        hero = html_escape(&hero), conf = best_conf * 100.0, rows = rows
    )
}

fn format_sigmoid(value: f64) -> String {
    let pct   = value * 100.0;
    let width = (value * 260.0) as u32;
    format!(
        r#"<div class="result-card"><h2>Result</h2>
<div class="prediction-hero">{:.4}</div>
<div class="prediction-sub">Output probability: {:.1}%</div>
<div style="margin-top:14px"><div class="bar-wrap" style="width:100%;max-width:300px"><div class="bar-fill" style="width:{}px"></div></div></div>
</div>"#,
        value, pct, width
    )
}

fn format_raw(output: &[f64]) -> String {
    let values: String = output.iter().enumerate()
        .map(|(i, v)| format!("[{}] {:.6}", i, v))
        .collect::<Vec<_>>().join("<br>");
    format!(
        r#"<div class="result-card"><h2>Result</h2><div class="raw-output">{}</div></div>"#,
        values
    )
}

fn error_html(msg: &str) -> String {
    format!(r#"<div class="result-card"><h2>Error</h2><div class="error-box">{}</div></div>"#, msg)
}

// ---------------------------------------------------------------------------
// POST /test/import-model
// ---------------------------------------------------------------------------

pub fn handle_import_model(request: &mut Request, state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let st   = state.lock().unwrap();
    let mask = st.tab_unlock_mask();
    drop(st);

    let content_type = request.headers().iter()
        .find(|h| h.field.equiv("Content-Type"))
        .map(|h| h.value.as_str().to_owned())
        .unwrap_or_default();

    let boundary = match extract_boundary(&content_type) {
        Some(b) => b,
        None    => {
            let page = build_test_page("", &error_html("Invalid multipart request."), mask);
            return crate::routes::html_response(page);
        }
    };

    let mut body: Vec<u8> = Vec::new();
    let _ = request.as_reader().read_to_end(&mut body);

    // Extract file bytes.
    let file_bytes = match multipart_extract_file(&body, &boundary) {
        Some(b) if !b.is_empty() => b,
        _ => {
            let page = build_test_page("", &error_html("No JSON file was uploaded."), mask);
            return crate::routes::html_response(page);
        }
    };

    // Basic JSON validation: must deserialize and contain a "layers" key.
    let json_val: serde_json::Value = match serde_json::from_slice(&file_bytes) {
        Ok(v)  => v,
        Err(_) => {
            let page = build_test_page("", &error_html("Uploaded file is not valid JSON."), mask);
            return crate::routes::html_response(page);
        }
    };
    if json_val.get("layers").is_none() {
        let page = build_test_page("", &error_html("JSON does not appear to be a Ferrite model (missing \"layers\" field)."), mask);
        return crate::routes::html_response(page);
    }

    // Extract the original filename from multipart headers.
    let raw_filename = extract_upload_filename(&body, &boundary)
        .unwrap_or_else(|| "imported_model".to_owned());

    // Strip path components and .json extension, then sanitize.
    let stem = std::path::Path::new(&raw_filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("imported_model");
    let sanitized: String = stem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    let model_name = if sanitized.is_empty() { "imported_model".to_owned() } else { sanitized };

    // Write to trained_models/.
    let model_dir  = "trained_models";
    let model_path = format!("{}/{}.json", model_dir, model_name);
    if let Err(_) = std::fs::create_dir_all(model_dir) {
        let page = build_test_page("", &error_html("Could not create trained_models/ directory."), mask);
        return crate::routes::html_response(page);
    }
    if let Err(_) = std::fs::write(&model_path, &file_bytes) {
        let page = build_test_page("", &error_html(&format!("Could not write model to '{}'.", model_path)), mask);
        return crate::routes::html_response(page);
    }

    // Redirect to /test?model=<name> so the new model is selected.
    crate::routes::redirect(&format!("/test?model={}", model_name))
}

/// Extracts the `filename="..."` value from the first file part of a multipart body.
fn extract_upload_filename(body: &[u8], boundary: &str) -> Option<String> {
    let delimiter = format!("--{}", boundary);
    let delim_bytes = delimiter.as_bytes();
    let parts = split_on(body, delim_bytes);

    for part in &parts {
        let sep = b"\r\n\r\n";
        if let Some(sep_pos) = find_subsequence(part, sep) {
            let header_section = &part[..sep_pos];
            let headers_str = String::from_utf8_lossy(header_section);
            // Only file parts have filename=.
            if !headers_str.contains("filename=") {
                continue;
            }
            // Parse filename="..." or filename=...
            let key = "filename=\"";
            if let Some(pos) = headers_str.find(key) {
                let rest = &headers_str[pos + key.len()..];
                if let Some(end) = rest.find('"') {
                    return Some(rest[..end].to_owned());
                }
            }
        }
    }
    None
}
