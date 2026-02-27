use std::io::Cursor;
use tiny_http::Response;

/// `GET /models/{name}/download`
///
/// Serves the JSON file for the named model as a downloadable attachment.
pub fn handle_download(name: &str) -> Response<Cursor<Vec<u8>>> {
    // Basic sanity check — reject empty names or path traversal attempts.
    if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
        return crate::routes::not_found();
    }

    let path = format!("trained_models/{}.json", name);
    match std::fs::read_to_string(&path) {
        Ok(json) => {
            let filename = format!("{}.json", name);
            crate::routes::json_download_response(json, &filename)
        }
        Err(_) => crate::routes::not_found(),
    }
}
