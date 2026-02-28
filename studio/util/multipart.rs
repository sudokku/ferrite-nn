/// Returns the index of the first occurrence of `needle` in `haystack`.
pub fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Splits `haystack` on every occurrence of `needle`, returning the pieces
/// between occurrences (excluding the needle itself).
pub fn split_on<'a>(haystack: &'a [u8], needle: &[u8]) -> Vec<&'a [u8]> {
    let mut result = Vec::new();
    let mut start = 0;
    while start <= haystack.len() {
        if let Some(pos) = find_subsequence(&haystack[start..], needle) {
            result.push(&haystack[start..start + pos]);
            start += pos + needle.len();
        } else {
            result.push(&haystack[start..]);
            break;
        }
    }
    result
}

/// Extracts the boundary token from a Content-Type header value like
/// `multipart/form-data; boundary=----WebKitFormBoundaryXXX`.
pub fn extract_boundary(content_type: &str) -> Option<String> {
    content_type
        .split(';')
        .map(|s| s.trim())
        .find(|s| s.starts_with("boundary="))
        .map(|s| s["boundary=".len()..].trim_matches('"').to_owned())
}

/// Extracts the raw bytes of the first file part from a multipart/form-data body.
/// Returns `None` if not found or on parse error.
pub fn multipart_extract_file(body: &[u8], boundary: &str) -> Option<Vec<u8>> {
    let delimiter = format!("--{}", boundary);
    let delim_bytes = delimiter.as_bytes();
    let parts = split_on(body, delim_bytes);

    for part in parts {
        let sep = b"\r\n\r\n";
        if let Some(sep_pos) = find_subsequence(part, sep) {
            let header_section = &part[..sep_pos];
            if header_section
                .windows(8)
                .any(|w| w.eq_ignore_ascii_case(b"filename"))
            {
                let data_start = sep_pos + sep.len();
                let raw = &part[data_start..];
                let trimmed = raw.strip_suffix(b"\r\n").unwrap_or(raw);
                return Some(trimmed.to_vec());
            }
        }
    }
    None
}

/// Extracts a plain-text (non-file) field from a multipart body.
pub fn extract_text_field(body: &[u8], boundary: &str, field_name: &str) -> Option<String> {
    let delimiter = format!("--{}", boundary);
    let delim_bytes = delimiter.as_bytes();
    let parts = split_on(body, delim_bytes);

    for part in parts {
        let sep = b"\r\n\r\n";
        if let Some(sep_pos) = find_subsequence(part, sep) {
            let header_section = &part[..sep_pos];
            let headers_str = String::from_utf8_lossy(header_section);
            let has_field = headers_str.contains(&format!("name=\"{}\"", field_name));
            let is_file   = headers_str.contains("filename=");
            if has_field && !is_file {
                let data_start = sep_pos + sep.len();
                let raw = &part[data_start..];
                let trimmed = raw.strip_suffix(b"\r\n").unwrap_or(raw);
                return String::from_utf8(trimmed.to_vec()).ok();
            }
        }
    }
    None
}

/// Extracts **all** text (non-file) fields from a multipart body as
/// `(name, value)` pairs.  Useful when iterating form fields generically.
pub fn extract_all_text_fields(body: &[u8], boundary: &str) -> Vec<(String, String)> {
    let delimiter = format!("--{}", boundary);
    let delim_bytes = delimiter.as_bytes();
    let parts = split_on(body, delim_bytes);
    let mut result = Vec::new();

    for part in parts {
        let sep = b"\r\n\r\n";
        if let Some(sep_pos) = find_subsequence(part, sep) {
            let header_section = &part[..sep_pos];
            let headers_str = String::from_utf8_lossy(header_section);
            // Only text fields (no filename=)
            if headers_str.contains("filename=") {
                continue;
            }
            // Extract name="..."
            if let Some(name) = parse_disposition_name(&headers_str) {
                let data_start = sep_pos + sep.len();
                let raw = &part[data_start..];
                let trimmed = raw.strip_suffix(b"\r\n").unwrap_or(raw);
                if let Ok(value) = String::from_utf8(trimmed.to_vec()) {
                    result.push((name, value));
                }
            }
        }
    }
    result
}

/// Extracts the raw bytes of a named file part from a multipart/form-data body.
///
/// Unlike `multipart_extract_file` which returns the first file encountered,
/// this function matches on the `name="<field_name>"` attribute so you can
/// pick a specific upload field when a form contains multiple file inputs.
pub fn multipart_extract_file_by_name(body: &[u8], boundary: &str, field_name: &str) -> Option<Vec<u8>> {
    let delimiter = format!("--{}", boundary);
    let delim_bytes = delimiter.as_bytes();
    let parts = split_on(body, delim_bytes);

    for part in parts {
        let sep = b"\r\n\r\n";
        if let Some(sep_pos) = find_subsequence(part, sep) {
            let header_section = &part[..sep_pos];
            let headers_str = String::from_utf8_lossy(header_section);
            let has_name     = headers_str.contains(&format!("name=\"{}\"", field_name));
            let has_filename = headers_str.contains("filename=");
            if has_name && has_filename {
                let data_start = sep_pos + sep.len();
                let raw = &part[data_start..];
                let trimmed = raw.strip_suffix(b"\r\n").unwrap_or(raw);
                return Some(trimmed.to_vec());
            }
        }
    }
    None
}

/// Parses the `name="..."` value from a Content-Disposition header string.
fn parse_disposition_name(headers: &str) -> Option<String> {
    let key = "name=\"";
    let pos = headers.find(key)?;
    let rest = &headers[pos + key.len()..];
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}
