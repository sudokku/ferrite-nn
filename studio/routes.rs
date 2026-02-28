use std::io::Cursor;
use tiny_http::{Header, Method, Request, Response, StatusCode};

use crate::state::SharedState;
use crate::handlers;

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

pub fn html_response(body: String) -> Response<Cursor<Vec<u8>>> {
    let bytes = body.into_bytes();
    let len = bytes.len();
    Response::new(
        StatusCode(200),
        vec![Header::from_bytes(b"Content-Type", b"text/html; charset=utf-8").unwrap()],
        Cursor::new(bytes),
        Some(len),
        None,
    )
}

pub fn redirect(location: &str) -> Response<Cursor<Vec<u8>>> {
    Response::new(
        StatusCode(303),
        vec![
            Header::from_bytes(b"Location", location.as_bytes()).unwrap(),
            Header::from_bytes(b"Content-Length", b"0").unwrap(),
        ],
        Cursor::new(Vec::new()),
        Some(0),
        None,
    )
}

pub fn json_download_response(body: String, filename: &str) -> Response<Cursor<Vec<u8>>> {
    let bytes = body.into_bytes();
    let len = bytes.len();
    let disposition = format!("attachment; filename=\"{}\"", filename);
    Response::new(
        StatusCode(200),
        vec![
            Header::from_bytes(b"Content-Type", b"application/json").unwrap(),
            Header::from_bytes(b"Content-Disposition", disposition.as_bytes()).unwrap(),
        ],
        Cursor::new(bytes),
        Some(len),
        None,
    )
}

pub fn not_found() -> Response<Cursor<Vec<u8>>> {
    let body = b"404 Not Found".to_vec();
    let len = body.len();
    Response::new(
        StatusCode(404),
        vec![Header::from_bytes(b"Content-Type", b"text/plain").unwrap()],
        Cursor::new(body),
        Some(len),
        None,
    )
}

// ---------------------------------------------------------------------------
// Request dispatcher
// ---------------------------------------------------------------------------

/// Dispatches incoming requests to the appropriate handler.
///
/// All handlers (except SSE) receive a `&mut Request` so that the dispatcher
/// retains ownership and can call `request.respond(response)` at the end.
/// The SSE handler takes ownership to perform long-lived streaming.
pub fn dispatch(mut request: Request, state: SharedState) {
    let method = request.method().clone();
    let url    = request.url().to_owned();

    let (path, query) = if let Some(pos) = url.find('?') {
        (url[..pos].to_owned(), url[pos + 1..].to_owned())
    } else {
        (url.clone(), String::new())
    };

    // SSE — long-lived; handler takes ownership and drives the stream loop.
    if method == Method::Get && path == "/train/events" {
        handlers::train_sse::handle(request, state);
        return;
    }

    // Model download — dynamic path segment.
    if method == Method::Get && path.starts_with("/models/") && path.ends_with("/download") {
        let name = path
            .strip_prefix("/models/")
            .and_then(|s| s.strip_suffix("/download"))
            .unwrap_or("")
            .to_owned();
        let resp = handlers::models::handle_download(&name);
        let _ = request.respond(resp);
        return;
    }

    let response = match (method, path.as_str()) {
        // ── Root redirect ─────────────────────────────────────────────────
        (Method::Get, "/") => redirect("/architect"),

        // ── Architect ────────────────────────────────────────────────────
        (Method::Get,  "/architect")       => handlers::architect::handle_get(state),
        (Method::Post, "/architect/save")  => handlers::architect::handle_post(&mut request, state),

        // ── Dataset ──────────────────────────────────────────────────────
        (Method::Get,  "/dataset")              => handlers::dataset::handle_get(state),
        (Method::Post, "/dataset/upload")       => handlers::dataset::handle_upload(&mut request, state),
        (Method::Post, "/dataset/upload-idx")   => handlers::dataset::handle_upload_idx(&mut request, state),
        (Method::Post, "/dataset/builtin")      => handlers::dataset::handle_builtin(&mut request, state),

        // ── Train ────────────────────────────────────────────────────────
        (Method::Get,  "/train")        => handlers::train::handle_get(state),
        (Method::Post, "/train/start")  => handlers::train::handle_start(state),
        (Method::Post, "/train/stop")   => handlers::train::handle_stop(state),

        // ── Evaluate ─────────────────────────────────────────────────────
        (Method::Get, "/evaluate")        => handlers::evaluate::handle_get(state),
        (Method::Get, "/evaluate/export") => handlers::evaluate::handle_export(state),

        // ── Test ─────────────────────────────────────────────────────────
        (Method::Get,  "/test")               => handlers::test::handle_get(query, state),
        (Method::Post, "/test/infer")         => handlers::test::handle_infer(&mut request, state),
        (Method::Post, "/test/import-model")  => handlers::test::handle_import_model(&mut request, state),

        // ── 404 ──────────────────────────────────────────────────────────
        _ => not_found(),
    };

    let _ = request.respond(response);
}
