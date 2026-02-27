/// CSV parsing utilities for the ferrite-nn studio.
///
/// Supported format:
/// - UTF-8, comma-separated
/// - Optional header row (auto-detected: first row is a header if it contains
///   any non-numeric, non-empty cell)
/// - Double-quoted fields with embedded commas are handled correctly
/// - Max upload size is enforced by the caller (50 MB)
///
/// Label modes:
/// - `ClassIndex` — the last column is an integer class index (0-based);
///   the server one-hot-encodes it into a vector of length `n_classes`.
/// - `OneHot`     — the last `n_classes` columns are floats forming the label.

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelMode {
    /// Last column is an integer class index; one-hot encode to `n_classes`.
    ClassIndex { n_classes: usize },
    /// Last `n_label_cols` columns are the label vector.
    OneHot { n_label_cols: usize },
}

#[derive(Debug)]
pub struct CsvParseError(pub String);

impl std::fmt::Display for CsvParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parses CSV bytes into (inputs, labels).
///
/// # Arguments
/// - `data`       — raw CSV bytes (UTF-8)
/// - `label_mode` — how to interpret the label column(s)
///
/// # Returns
/// `(inputs, labels)` where each is a `Vec<Vec<f64>>` of equal length.
pub fn parse_csv(
    data: &[u8],
    label_mode: LabelMode,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), CsvParseError> {
    let text = std::str::from_utf8(data)
        .map_err(|_| CsvParseError("CSV file is not valid UTF-8".into()))?;

    let mut lines = text.lines().peekable();

    // Auto-detect header: skip first line if any cell is non-numeric.
    if let Some(first) = lines.peek() {
        if is_header(first) {
            lines.next();
        }
    }

    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<Vec<f64>> = Vec::new();

    for (row_idx, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let cells = parse_csv_row(line);
        if cells.is_empty() {
            continue;
        }

        match label_mode {
            LabelMode::ClassIndex { n_classes } => {
                if cells.len() < 2 {
                    return Err(CsvParseError(format!(
                        "Row {}: expected at least 2 columns (features + class index), got {}",
                        row_idx + 1,
                        cells.len()
                    )));
                }
                let feature_cells = &cells[..cells.len() - 1];
                let label_cell    = cells.last().unwrap();

                let feats = parse_floats(feature_cells, row_idx + 1)?;
                let class_idx: usize = label_cell.trim().parse::<usize>().map_err(|_| {
                    CsvParseError(format!(
                        "Row {}: class index '{}' is not a non-negative integer",
                        row_idx + 1,
                        label_cell
                    ))
                })?;
                if class_idx >= n_classes {
                    return Err(CsvParseError(format!(
                        "Row {}: class index {} >= n_classes {}",
                        row_idx + 1, class_idx, n_classes
                    )));
                }
                let mut one_hot = vec![0.0f64; n_classes];
                one_hot[class_idx] = 1.0;

                inputs.push(feats);
                labels.push(one_hot);
            }
            LabelMode::OneHot { n_label_cols } => {
                if cells.len() < n_label_cols + 1 {
                    return Err(CsvParseError(format!(
                        "Row {}: expected at least {} columns, got {}",
                        row_idx + 1,
                        n_label_cols + 1,
                        cells.len()
                    )));
                }
                let split = cells.len() - n_label_cols;
                let feature_cells = &cells[..split];
                let label_cells   = &cells[split..];

                let feats  = parse_floats(feature_cells, row_idx + 1)?;
                let lbls   = parse_floats(label_cells,   row_idx + 1)?;

                inputs.push(feats);
                labels.push(lbls);
            }
        }
    }

    if inputs.is_empty() {
        return Err(CsvParseError("CSV contains no data rows after parsing".into()));
    }

    // Verify all rows have the same feature width.
    let n_feats = inputs[0].len();
    for (i, row) in inputs.iter().enumerate() {
        if row.len() != n_feats {
            return Err(CsvParseError(format!(
                "Row {}: feature count {} does not match first row's {}",
                i + 1, row.len(), n_feats
            )));
        }
    }

    Ok((inputs, labels))
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Returns `true` if the row looks like a header (any cell non-numeric).
fn is_header(line: &str) -> bool {
    let cells = parse_csv_row(line);
    cells.iter().any(|c| {
        let t = c.trim();
        !t.is_empty() && t.parse::<f64>().is_err()
    })
}

/// Parses a single CSV row, handling double-quoted fields.
fn parse_csv_row(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '"' => {
                if in_quotes && i + 1 < chars.len() && chars[i + 1] == '"' {
                    // Escaped quote inside quoted field.
                    current.push('"');
                    i += 2;
                    continue;
                }
                in_quotes = !in_quotes;
            }
            ',' if !in_quotes => {
                fields.push(current.clone());
                current.clear();
            }
            c => current.push(c),
        }
        i += 1;
    }
    fields.push(current);
    fields
}

/// Parses a slice of string cells as `f64`, returning an error with row info on failure.
fn parse_floats(cells: &[String], row_num: usize) -> Result<Vec<f64>, CsvParseError> {
    cells.iter()
        .map(|c| {
            c.trim().parse::<f64>().map_err(|_| {
                CsvParseError(format!(
                    "Row {}: '{}' is not a valid number",
                    row_num, c
                ))
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Built-in toy datasets
// ---------------------------------------------------------------------------

/// Returns the XOR dataset: 4 samples, 2 inputs, 1 one-hot output (2 classes).
pub fn builtin_xor() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let labels = vec![
        vec![1.0, 0.0], // XOR = 0
        vec![0.0, 1.0], // XOR = 1
        vec![0.0, 1.0], // XOR = 1
        vec![1.0, 0.0], // XOR = 0
    ];
    (inputs, labels)
}

/// Generates `n` samples of 2D "two circles" data (class 0 = inner, class 1 = outer).
/// Outputs are one-hot vectors of length 2.
pub fn builtin_circles(n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    use std::f64::consts::PI;
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let class = i % 2;
        let angle = (i as f64 / n as f64) * 2.0 * PI * 10.0;
        let radius = if class == 0 { 0.3 } else { 0.8 };
        // Add small deterministic "noise" via a second sinusoidal.
        let noise = 0.05 * ((i as f64 * 7.3).sin());
        let x = (radius + noise) * angle.cos();
        let y = (radius + noise) * angle.sin();
        // Normalize to [0, 1].
        inputs.push(vec![(x + 1.0) / 2.0, (y + 1.0) / 2.0]);
        let mut oh = vec![0.0, 0.0];
        oh[class] = 1.0;
        labels.push(oh);
    }
    (inputs, labels)
}

/// Generates `n` samples of 2D "two blobs" data.
/// Outputs are one-hot vectors of length 2.
pub fn builtin_blobs(n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    // Centers: class 0 at (0.3, 0.3), class 1 at (0.7, 0.7).
    let centers = [(0.3f64, 0.3f64), (0.7f64, 0.7f64)];
    for i in 0..n {
        let class = i % 2;
        let (cx, cy) = centers[class];
        // Deterministic "pseudo-random" spread using sin/cos of index.
        let angle = i as f64 * 2.399; // irrational-ish step
        let r = 0.12 * (i as f64 * 0.31).sin().abs();
        let x = (cx + r * angle.cos()).clamp(0.0, 1.0);
        let y = (cy + r * angle.sin()).clamp(0.0, 1.0);
        inputs.push(vec![x, y]);
        let mut oh = vec![0.0, 0.0];
        oh[class] = 1.0;
        labels.push(oh);
    }
    (inputs, labels)
}
