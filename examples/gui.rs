/// ferrite-nn · web inference GUI
///
/// A minimal synchronous HTTP server that lets you load any pretrained
/// ferrite-nn model (JSON) and run inference directly in your browser.
///
/// Run with:
///   cargo run --example gui --release
/// Then open http://127.0.0.1:7878

use std::fs;
use std::io::Cursor;
use tiny_http::{Header, Method, Response, Server};

use ferrite_nn::{ActivationFunction, Network};

// The HTML template is embedded at compile time so the binary is fully
// self-contained (no runtime file reads, works from any working directory).
const TEMPLATE: &str = include_str!("gui/index.html");

// ---------------------------------------------------------------------------
// URL / form-body helpers
// ---------------------------------------------------------------------------

/// Decodes a percent-encoded string (`%XX`) and converts `+` to space.
/// Handles malformed sequences gracefully (leaves them as-is).
fn url_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = (bytes[i + 1] as char).to_digit(16);
                let lo = (bytes[i + 2] as char).to_digit(16);
                match (hi, lo) {
                    (Some(h), Some(l)) => {
                        out.push((((h << 4) | l) as u8) as char);
                        i += 3;
                    }
                    _ => {
                        out.push('%');
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b as char);
                i += 1;
            }
        }
    }
    out
}

/// Parses `key=value&key2=value2` into a Vec of (key, value) pairs.
fn parse_form(body: &str) -> Vec<(String, String)> {
    body.split('&')
        .filter_map(|pair| {
            let mut it = pair.splitn(2, '=');
            let k = it.next()?.to_owned();
            let v = it.next().unwrap_or("").to_owned();
            Some((url_decode(&k), url_decode(&v)))
        })
        .collect()
}

/// Looks up a key in the parsed form pairs.
fn form_get<'a>(pairs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

// ---------------------------------------------------------------------------
// Model directory scanner
// ---------------------------------------------------------------------------

/// Returns the stem names (without extension) of all *.json files found in
/// `examples/trained_models/`, sorted alphabetically.
fn list_models() -> Vec<String> {
    let dir = "examples/trained_models";
    match fs::read_dir(dir) {
        Ok(entries) => {
            let mut names: Vec<String> = entries
                .flatten()
                .filter_map(|e| {
                    let path = e.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json") {
                        path.file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_owned())
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

/// Builds the `<option>` HTML for the model selector.
fn build_model_options(models: &[String], selected: &str) -> String {
    if models.is_empty() {
        return "<option disabled>No models found in examples/trained_models/</option>".into();
    }
    models
        .iter()
        .map(|name| {
            let sel = if name == selected { " selected" } else { "" };
            format!("<option value=\"{}\"{}>  {}</option>", name, sel, name)
        })
        .collect::<Vec<_>>()
        .join("\n        ")
}

// ---------------------------------------------------------------------------
// Inference & output formatting
// ---------------------------------------------------------------------------

/// Runs inference and returns an HTML snippet describing the result.
fn run_inference(model_name: &str, raw_inputs: &str) -> String {
    let path = format!("examples/trained_models/{}.json", model_name);

    // Load model
    let mut network = match Network::load_json(&path) {
        Ok(n) => n,
        Err(e) => {
            return error_html(&format!(
                "Could not load model <strong>{}</strong>: {}",
                model_name, e
            ))
        }
    };

    if network.layers.is_empty() {
        return error_html("Model has no layers.");
    }

    // Parse inputs
    let inputs: Vec<f64> = raw_inputs
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();

    let expected_len = network.layers[0].weights.rows;
    if inputs.len() != expected_len {
        return error_html(&format!(
            "Input length mismatch: model expects <strong>{}</strong> values, \
             but <strong>{}</strong> were provided.",
            expected_len,
            inputs.len()
        ));
    }

    // Forward pass
    let output = network.forward(inputs);

    // Format result based on output layer activator
    let last = network.layers.last().unwrap();
    match &last.activator {
        ActivationFunction::Softmax => format_softmax(&output),
        ActivationFunction::Sigmoid if output.len() == 1 => format_sigmoid(output[0]),
        _ => format_raw(&output),
    }
}

/// Renders a Softmax output as predicted class + probability table.
fn format_softmax(output: &[f64]) -> String {
    let n = output.len();
    // argmax
    let (best_class, best_conf) = output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));

    // Sorted indices by confidence descending
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| output[b].partial_cmp(&output[a]).unwrap());

    let rows: String = sorted
        .iter()
        .map(|&i| {
            let pct = output[i] * 100.0;
            let width = (output[i] * 260.0) as u32;
            let dim = if i != best_class { " dim" } else { "" };
            format!(
                "<tr>\
                  <td style=\"width:40px;font-weight:600;color:#333\">{}</td>\
                  <td><div class=\"bar-wrap\"><div class=\"bar-fill{}\" style=\"width:{}px\"></div></div></td>\
                  <td class=\"prob-pct\">{:.1}%</td>\
                </tr>",
                i, dim, width, pct
            )
        })
        .collect();

    format!(
        "<div class=\"result-card\">\
          <h2>✅ Result</h2>\
          <div class=\"prediction-hero\">Class {}</div>\
          <div class=\"prediction-sub\">Confidence: {:.1}%</div>\
          <table class=\"prob-table\">\
            <thead><tr>\
              <th>Class</th>\
              <th>Confidence</th>\
              <th></th>\
            </tr></thead>\
            <tbody>{}</tbody>\
          </table>\
        </div>",
        best_class,
        best_conf * 100.0,
        rows
    )
}

/// Renders a single Sigmoid output.
fn format_sigmoid(value: f64) -> String {
    let pct = value * 100.0;
    let width = (value * 260.0) as u32;
    format!(
        "<div class=\"result-card\">\
          <h2>✅ Result</h2>\
          <div class=\"prediction-hero\">{:.4}</div>\
          <div class=\"prediction-sub\">Output probability: {:.1}%</div>\
          <div style=\"margin-top:14px\">\
            <div class=\"bar-wrap\" style=\"width:100%;max-width:300px\">\
              <div class=\"bar-fill\" style=\"width:{}px\"></div>\
            </div>\
          </div>\
        </div>",
        value, pct, width
    )
}

/// Renders raw output values for Identity / other activators.
fn format_raw(output: &[f64]) -> String {
    let values: String = output
        .iter()
        .enumerate()
        .map(|(i, v)| format!("[{}] {:.6}", i, v))
        .collect::<Vec<_>>()
        .join("<br>");
    format!(
        "<div class=\"result-card\">\
          <h2>✅ Result</h2>\
          <div class=\"raw-output\">{}</div>\
        </div>",
        values
    )
}

/// Renders a user-facing error block.
fn error_html(msg: &str) -> String {
    format!(
        "<div class=\"result-card\">\
          <h2>⚠️ Error</h2>\
          <div class=\"error-box\">{}</div>\
        </div>",
        msg
    )
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

fn html_response(body: String) -> Response<Cursor<Vec<u8>>> {
    let bytes = body.into_bytes();
    let len = bytes.len();
    Response::new(
        tiny_http::StatusCode(200),
        vec![Header::from_bytes(b"Content-Type", b"text/html; charset=utf-8").unwrap()],
        Cursor::new(bytes),
        Some(len),
        None,
    )
}

fn not_found() -> Response<Cursor<Vec<u8>>> {
    let body = b"404 Not Found".to_vec();
    let len = body.len();
    Response::new(
        tiny_http::StatusCode(404),
        vec![Header::from_bytes(b"Content-Type", b"text/plain").unwrap()],
        Cursor::new(body),
        Some(len),
        None,
    )
}

// ---------------------------------------------------------------------------
// Page builder
// ---------------------------------------------------------------------------

fn render_page(model_options: &str, result_section: &str, input_values: &str) -> String {
    TEMPLATE
        .replace("{{MODEL_OPTIONS}}", model_options)
        .replace("{{RESULT_SECTION}}", result_section)
        .replace("{{INPUT_VALUES}}", input_values)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let addr = "127.0.0.1:7878";
    let server = Server::http(addr).expect("Failed to bind HTTP server");

    println!("╔══════════════════════════════════════════╗");
    println!("║       ferrite-nn · inference GUI         ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Open in your browser:                   ║");
    println!("║  http://{}               ║", addr);
    println!("╠══════════════════════════════════════════╣");

    let models = list_models();
    if models.is_empty() {
        println!("║  ⚠  No models found in                  ║");
        println!("║     examples/trained_models/             ║");
        println!("║  Train one first:                        ║");
        println!("║  cargo run --example mnist --release     ║");
    } else {
        println!("║  Models loaded ({:>2}):                    ║", models.len());
        for m in &models {
            println!("║    • {:<37}║", m);
        }
    }
    println!("╚══════════════════════════════════════════╝");

    for mut request in server.incoming_requests() {
        let method = request.method().clone();
        let url = request.url().to_owned();

        let response = match (method, url.as_str()) {
            // ── GET / ──────────────────────────────────────────────────────
            (Method::Get, "/") => {
                let models = list_models();
                let options = build_model_options(&models, models.first().map(|s| s.as_str()).unwrap_or(""));
                html_response(render_page(&options, "", ""))
            }

            // ── POST /infer ───────────────────────────────────────────────
            (Method::Post, "/infer") => {
                // Read body
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);

                let pairs = parse_form(&body);
                let model_name = form_get(&pairs, "model").unwrap_or("").to_owned();
                let raw_inputs = form_get(&pairs, "inputs").unwrap_or("").to_owned();

                let models = list_models();
                let options = build_model_options(&models, &model_name);
                let result = run_inference(&model_name, &raw_inputs);

                html_response(render_page(&options, &result, &raw_inputs))
            }

            // ── 404 ───────────────────────────────────────────────────────
            _ => not_found(),
        };

        let _ = request.respond(response);
    }
}
