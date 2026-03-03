/// Decodes a percent-encoded string (`%XX`) and converts `+` to space.
pub fn url_decode(s: &str) -> String {
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

/// Parses `key=value&key2=value2` into a `Vec` of `(key, value)` pairs.
pub fn parse_form(body: &str) -> Vec<(String, String)> {
    body.split('&')
        .filter_map(|pair| {
            let mut it = pair.splitn(2, '=');
            let k = it.next()?.to_owned();
            let v = it.next().unwrap_or("").to_owned();
            Some((url_decode(&k), url_decode(&v)))
        })
        .collect()
}

/// Looks up a key in parsed form pairs, returning the value if found.
pub fn form_get<'a>(pairs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}
