use tiny_http::{Request, Response};
use std::io::Cursor;

use ferrite_nn::{ActivationFunction, LossType, NetworkSpec, LayerSpec};

use crate::state::{FlashMessage, Hyperparams, SharedState, TrainingStatus};
use crate::util::form::{parse_form, form_get};
use crate::render::{render_page, Page};

// ---------------------------------------------------------------------------
// GET /architect
// ---------------------------------------------------------------------------

pub fn handle_get(state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let mut st = state.lock().unwrap();
    let flash = st.take_flash();
    let tab_unlock = st.tab_unlock_mask();
    let spec       = st.spec.clone();
    let hyperparams = st.hyperparams.clone();
    drop(st);

    let page = build_arch_page(&spec, &hyperparams, None, flash, tab_unlock);
    crate::routes::html_response(page)
}

// ---------------------------------------------------------------------------
// POST /architect/save
// ---------------------------------------------------------------------------

pub fn handle_post(request: &mut Request, state: SharedState) -> Response<Cursor<Vec<u8>>> {
    let mut body = String::new();
    let _ = request.as_reader().read_to_string(&mut body);
    let pairs = parse_form(&body);

    let name         = form_get(&pairs, "name").unwrap_or("").trim().to_owned();
    let description  = form_get(&pairs, "description").unwrap_or("").trim().to_owned();
    let input_size_s = form_get(&pairs, "input_size").unwrap_or("1").to_owned();
    let loss_s       = form_get(&pairs, "loss_type").unwrap_or("mse").to_owned();
    let lr_s         = form_get(&pairs, "learning_rate").unwrap_or("0.01").to_owned();
    let bs_s         = form_get(&pairs, "batch_size").unwrap_or("32").to_owned();
    let ep_s         = form_get(&pairs, "epochs").unwrap_or("50").to_owned();
    let layers_json  = form_get(&pairs, "layers_json").unwrap_or("[]").to_owned();

    // Helper: return error page using current state as defaults.
    let show_err = |err: &str, state: &SharedState| -> Response<Cursor<Vec<u8>>> {
        let st = state.lock().unwrap();
        let mask = st.tab_unlock_mask();
        let spec = st.spec.clone();
        let hp   = st.hyperparams.clone();
        drop(st);
        crate::routes::html_response(build_arch_page(&spec, &hp, Some(err), None, mask))
    };

    if name.is_empty() {
        return show_err("Model name must not be empty.", &state);
    }

    let input_size: usize = match input_size_s.trim().parse() {
        Ok(v) if v > 0 => v,
        _ => return show_err("Input size must be a positive integer.", &state),
    };

    let lr: f64 = match lr_s.trim().parse::<f64>() {
        Ok(v) if v > 0.0 => v,
        _ => return show_err("Learning rate must be a positive number.", &state),
    };

    let bs: usize = match bs_s.trim().parse() {
        Ok(v) if v > 0 => v,
        _ => return show_err("Batch size must be a positive integer.", &state),
    };

    let ep: usize = match ep_s.trim().parse() {
        Ok(v) if v > 0 => v,
        _ => return show_err("Epochs must be a positive integer.", &state),
    };

    // Parse layers JSON (sent by the JS prepareSubmit() function).
    #[derive(serde::Deserialize)]
    struct RawLayer { neurons: usize, activation: String }

    let raw_layers: Vec<RawLayer> = match serde_json::from_str(&layers_json) {
        Ok(v) => v,
        Err(_) => return show_err("Could not parse layer definitions.", &state),
    };

    if raw_layers.is_empty() {
        return show_err("Add at least one layer.", &state);
    }

    for rl in &raw_layers {
        if rl.neurons == 0 {
            return show_err("Each layer must have at least 1 neuron.", &state);
        }
    }

    // Build LayerSpec list.
    let mut layer_specs: Vec<LayerSpec> = Vec::new();
    let mut prev_size = input_size;
    for rl in &raw_layers {
        let activation = parse_activation(&rl.activation);
        layer_specs.push(LayerSpec { size: rl.neurons, input_size: prev_size, activation });
        prev_size = rl.neurons;
    }

    let loss = if loss_s == "cross_entropy" { LossType::CrossEntropy } else { LossType::Mse };

    // Enforce Softmax <-> CrossEntropy consistency.
    let last_act = &layer_specs.last().unwrap().activation;
    if *last_act == ActivationFunction::Softmax && loss != LossType::CrossEntropy {
        return show_err(
            "Softmax output requires Cross-Entropy loss. Please change the loss function.",
            &state,
        );
    }
    if *last_act != ActivationFunction::Softmax && loss == LossType::CrossEntropy {
        return show_err(
            "Cross-Entropy loss requires a Softmax output layer.",
            &state,
        );
    }

    let mut spec = NetworkSpec { name: name.clone(), layers: layer_specs, loss, metadata: None };
    if !description.is_empty() {
        spec.metadata = Some(ferrite_nn::ModelMetadata {
            description: Some(description),
            input_type:  None,
            output_labels: None,
        });
    }

    let hyperparams = Hyperparams { learning_rate: lr, batch_size: bs, epochs: ep };

    let mut st = state.lock().unwrap();
    st.spec        = Some(spec);
    st.hyperparams = Some(hyperparams);
    // Clear stale state when the architecture changes.
    st.dataset         = None;
    st.epoch_history.clear();
    st.trained_network = None;
    st.training        = TrainingStatus::Idle;
    st.flash = Some(FlashMessage::success(
        format!("Architecture '{}' saved successfully.", name)
    ));
    drop(st);

    crate::routes::redirect("/architect")
}

// ---------------------------------------------------------------------------
// Page builder
// ---------------------------------------------------------------------------

fn build_arch_page(
    spec: &Option<NetworkSpec>,
    hyperparams: &Option<Hyperparams>,
    error: Option<&str>,
    flash: Option<FlashMessage>,
    tab_unlock: u8,
) -> String {
    let name       = spec.as_ref().map(|s| s.name.as_str()).unwrap_or("");
    let desc       = spec.as_ref()
        .and_then(|s| s.metadata.as_ref())
        .and_then(|m| m.description.as_deref())
        .unwrap_or("");
    let input_size = spec.as_ref()
        .and_then(|s| s.layers.first())
        .map(|l| l.input_size)
        .unwrap_or(2);
    let loss       = spec.as_ref().map(|s| s.loss).unwrap_or(LossType::Mse);
    let lr         = hyperparams.as_ref().map(|h| h.learning_rate).unwrap_or(0.01);
    let bs         = hyperparams.as_ref().map(|h| h.batch_size).unwrap_or(32);
    let ep         = hyperparams.as_ref().map(|h| h.epochs).unwrap_or(50);

    let layer_rows = spec.as_ref()
        .map(|s| build_layer_rows(&s.layers))
        .unwrap_or_else(default_layer_rows);

    let flash_html = render_flash_html(flash.as_ref());
    let error_html = error.map(|e| {
        format!(r#"<div class="flash flash-error" style="margin-top:14px">{}</div>"#,
                html_escape(e))
    }).unwrap_or_default();

    let sel_mse = if loss == LossType::Mse { " selected" } else { "" };
    let sel_ce  = if loss == LossType::CrossEntropy { " selected" } else { "" };

    render_page(Page::Architect, tab_unlock, false, |tmpl| {
        tmpl
            .replace("{{FLASH_ARCH}}", &flash_html)
            .replace("{{ARCH_NAME}}", &html_escape(name))
            .replace("{{ARCH_DESC}}", &html_escape(desc))
            .replace("{{ARCH_INPUT_SIZE}}", &input_size.to_string())
            .replace("{{LAYER_ROWS}}", &layer_rows)
            .replace("{{SEL_MSE}}", sel_mse)
            .replace("{{SEL_CE}}", sel_ce)
            .replace("{{ARCH_LR}}", &lr.to_string())
            .replace("{{ARCH_BS}}", &bs.to_string())
            .replace("{{ARCH_EP}}", &ep.to_string())
            .replace("{{ARCH_ERROR}}", &error_html)
    })
}

fn build_layer_rows(layers: &[LayerSpec]) -> String {
    layers.iter().enumerate().map(|(i, ls)| {
        let idx     = i + 1;
        let act_str = activation_to_str(&ls.activation);
        let opts: String = ["sigmoid","relu","identity","softmax"].iter().map(|&a| {
            let sel = if a == act_str { " selected" } else { "" };
            let label = a[..1].to_uppercase() + &a[1..];
            format!("<option value=\"{}\"{}>{}</option>", a, sel, label)
        }).collect();
        format!(
            r#"<tr id="lr-{idx}"><td>{idx}</td><td><input type="number" class="neurons-input" data-field="neurons" value="{sz}" min="1"></td><td><select class="act-select" data-field="activation">{opts}</select></td><td><button type="button" class="btn btn-secondary btn-sm" onclick="removeLayer({idx})">Remove</button></td></tr>"#,
            idx = idx, sz = ls.size, opts = opts
        )
    }).collect::<Vec<_>>().join("\n")
}

fn default_layer_rows() -> String {
    r#"<tr id="lr-1"><td>1</td><td><input type="number" class="neurons-input" data-field="neurons" value="8" min="1"></td><td><select class="act-select" data-field="activation"><option value="sigmoid">Sigmoid</option><option value="relu" selected>Relu</option><option value="identity">Identity</option><option value="softmax">Softmax</option></select></td><td><button type="button" class="btn btn-secondary btn-sm" onclick="removeLayer(1)">Remove</button></td></tr>
<tr id="lr-2"><td>2</td><td><input type="number" class="neurons-input" data-field="neurons" value="2" min="1"></td><td><select class="act-select" data-field="activation"><option value="sigmoid">Sigmoid</option><option value="relu">Relu</option><option value="identity">Identity</option><option value="softmax" selected>Softmax</option></select></td><td><button type="button" class="btn btn-secondary btn-sm" onclick="removeLayer(2)">Remove</button></td></tr>"#.to_owned()
}

// ---------------------------------------------------------------------------
// Shared helpers (also used by other handlers)
// ---------------------------------------------------------------------------

pub fn parse_activation(s: &str) -> ActivationFunction {
    match s {
        "relu"     => ActivationFunction::ReLU,
        "softmax"  => ActivationFunction::Softmax,
        "identity" => ActivationFunction::Identity,
        _          => ActivationFunction::Sigmoid,
    }
}

pub fn activation_to_str(a: &ActivationFunction) -> &'static str {
    match a {
        ActivationFunction::ReLU     => "relu",
        ActivationFunction::Softmax  => "softmax",
        ActivationFunction::Identity => "identity",
        ActivationFunction::Sigmoid  => "sigmoid",
    }
}

pub fn render_flash_html(flash: Option<&FlashMessage>) -> String {
    match flash {
        None    => String::new(),
        Some(f) => {
            let cls = match f.kind {
                crate::state::FlashKind::Success => "flash-success",
                crate::state::FlashKind::Error   => "flash-error",
            };
            format!(r#"<div class="flash {}">{}</div>"#, cls, html_escape(&f.text))
        }
    }
}

pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
}
