#![allow(dead_code)]
use std::io::Write;
use tiny_http::{Header, Response};

// ---------------------------------------------------------------------------
// SSE response helpers
// ---------------------------------------------------------------------------

/// Formats a named SSE event with a JSON data payload.
///
/// Output format (per SSE spec):
/// ```
/// event: <name>\n
/// data: <json>\n
/// \n
/// ```
pub fn format_sse_event(event_name: &str, json_data: &str) -> String {
    format!("event: {}\ndata: {}\n\n", event_name, json_data)
}

/// Formats a keep-alive SSE comment.
/// SSE comments start with `:` and are ignored by EventSource clients
/// but prevent the connection from timing out.
pub fn format_sse_keepalive() -> &'static str {
    ": ping\n\n"
}

/// Writes a single SSE message to a writer, flushing immediately.
/// Returns `false` if the write failed (client disconnected).
pub fn write_sse<W: Write>(writer: &mut W, msg: &str) -> bool {
    writer.write_all(msg.as_bytes()).is_ok() && writer.flush().is_ok()
}

/// Creates the HTTP response headers for an SSE stream.
/// The body must be written externally via the raw writer.
pub fn sse_headers() -> Vec<Header> {
    vec![
        Header::from_bytes(b"Content-Type", b"text/event-stream").unwrap(),
        Header::from_bytes(b"Cache-Control", b"no-cache").unwrap(),
        Header::from_bytes(b"Connection", b"keep-alive").unwrap(),
        Header::from_bytes(b"X-Accel-Buffering", b"no").unwrap(),
    ]
}

/// Builds a complete SSE response with an empty body (used to hand off to
/// `respond_with_writer`).  Since tiny_http doesn't support streaming bodies
/// natively, we use `Response::new` with a `Cursor` and write SSE frames
/// via `request.as_writer()` after consuming the initial response.
///
/// In practice the studio's SSE handler uses `request.as_writer()` directly
/// after sending the headers with a zero-length initial body.
pub fn build_sse_response() -> Response<std::io::Cursor<Vec<u8>>> {
    let body: Vec<u8> = Vec::new();
    Response::new(
        tiny_http::StatusCode(200),
        sse_headers(),
        std::io::Cursor::new(body),
        Some(0),
        None,
    )
}
