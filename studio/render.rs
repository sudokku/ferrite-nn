/// Central template renderer for the ferrite-nn studio.
///
/// The studio uses a single HTML template (`studio/assets/studio.html`) with
/// placeholder tokens like `{{TOKEN}}`.  This module loads the template at
/// compile time and exposes a single `render_page` function that accepts a
/// closure to do tab-specific placeholder substitution.
///
/// Placeholders that are global across all pages (tab_unlock, active_tab,
/// training_running) are resolved here before calling the closure; tab-specific
/// placeholders that were not replaced by the closure are blanked to avoid
/// leaking raw `{{TOKEN}}` strings to the browser.

const TEMPLATE: &str = include_str!("assets/studio.html");

/// Which tab is active — controls both the active CSS class and the JS
/// `ACTIVE_TAB` variable injected into the page.
#[derive(Clone, Copy)]
pub enum Page {
    Architect = 0,
    Dataset   = 1,
    Train     = 2,
    Evaluate  = 3,
    Test      = 4,
}

/// Renders the full studio page.
///
/// # Arguments
/// - `page`             — active tab index
/// - `tab_unlock`       — bitmask; see `StudioState::tab_unlock_mask()`
/// - `training_running` — whether a training job is currently active
/// - `fill`             — closure that fills tab-specific placeholders
pub fn render_page<F>(page: Page, tab_unlock: u8, training_running: bool, fill: F) -> String
where
    F: FnOnce(String) -> String,
{
    let mut html = TEMPLATE.to_owned();

    // Inject global JS variables.
    html = html.replace("{{TAB_UNLOCK}}",      &tab_unlock.to_string());
    html = html.replace("{{ACTIVE_TAB}}",      &(page as u8).to_string());
    html = html.replace("{{TRAINING_RUNNING}}", if training_running { "true" } else { "false" });

    // Let the caller fill tab-specific placeholders.
    html = fill(html);

    // Blank any remaining unfilled placeholders (prevents raw `{{TOKEN}}` in output).
    blank_remaining(html)
}

/// Replaces any `{{UPPERCASE_TOKEN}}` that wasn't already substituted with an
/// empty string.  This is a safety net — all tokens should be handled by the
/// caller, but a missed token should produce a clean page rather than leaking
/// debug info.
fn blank_remaining(mut html: String) -> String {
    // We loop because a single pass with a start-cursor approach is cleanest.
    while let Some(start) = html.find("{{") {
        if let Some(end) = html[start..].find("}}") {
            let abs_end = start + end + 2;
            html.replace_range(start..abs_end, "");
        } else {
            break;
        }
    }
    html
}
