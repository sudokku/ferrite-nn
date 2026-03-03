/// Parse a pair of IDX binary files (image + label) as used by MNIST and its
/// derivatives (Fashion-MNIST, EMNIST, …) into `(inputs, labels)` suitable for
/// direct use with Ferrite's training loop.
///
/// # IDX3 image file layout
/// ```text
/// bytes  0-1:   0x00 0x00   (reserved, must be zero)
/// byte   2:     0x08        (dtype = uint8)
/// byte   3:     0x03        (number of dimensions = 3)
/// bytes  4-7:   N           (number of images, big-endian u32)
/// bytes  8-11:  rows        (image height in pixels, big-endian u32)
/// bytes 12-15:  cols        (image width in pixels, big-endian u32)
/// bytes 16..:   N * rows * cols bytes, row-major, uint8
/// ```
///
/// # IDX1 label file layout
/// ```text
/// bytes  0-1:   0x00 0x00   (reserved, must be zero)
/// byte   2:     0x08        (dtype = uint8)
/// byte   3:     0x01        (number of dimensions = 1)
/// bytes  4-7:   N           (number of labels, big-endian u32)
/// bytes  8..:   N bytes, each a class index in [0, n_classes)
/// ```
///
/// # Returns
/// `(inputs, labels)` where
/// - `inputs[i]`  is a `Vec<f64>` of length `rows * cols`, each pixel divided
///   by 255.0 so values lie in `[0.0, 1.0]`.
/// - `labels[i]`  is a one-hot `Vec<f64>` of length `n_classes`.
pub fn parse_idx_pair(
    image_bytes: &[u8],
    label_bytes: &[u8],
    n_classes: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
    // ── Image file validation ───────────────────────────────────────────────

    if image_bytes.len() < 16 {
        return Err(format!(
            "IDX image file too short: expected at least 16 header bytes, got {}.",
            image_bytes.len()
        ));
    }

    if image_bytes[0] != 0x00 || image_bytes[1] != 0x00 {
        return Err(format!(
            "IDX image file: bytes 0-1 must be 0x00 0x00 (reserved), got 0x{:02X} 0x{:02X}.",
            image_bytes[0], image_bytes[1]
        ));
    }
    if image_bytes[2] != 0x08 {
        return Err(format!(
            "IDX image file: byte 2 (dtype) must be 0x08 (uint8), got 0x{:02X}.",
            image_bytes[2]
        ));
    }
    if image_bytes[3] != 0x03 {
        return Err(format!(
            "IDX image file: byte 3 (dimensions) must be 3, got {}. \
             This does not appear to be an IDX3 image file.",
            image_bytes[3]
        ));
    }

    let n_items = u32::from_be_bytes([
        image_bytes[4], image_bytes[5], image_bytes[6], image_bytes[7],
    ]) as usize;
    let rows = u32::from_be_bytes([
        image_bytes[8], image_bytes[9], image_bytes[10], image_bytes[11],
    ]) as usize;
    let cols = u32::from_be_bytes([
        image_bytes[12], image_bytes[13], image_bytes[14], image_bytes[15],
    ]) as usize;

    let n_pixels = rows.checked_mul(cols).ok_or_else(|| {
        format!("IDX image file: rows * cols overflows usize (rows={}, cols={}).", rows, cols)
    })?;
    let required_image_len = 16_usize
        .checked_add(n_items.checked_mul(n_pixels).ok_or_else(|| {
            format!(
                "IDX image file: n_items * n_pixels overflows usize \
                 (n_items={}, n_pixels={}).",
                n_items, n_pixels
            )
        })?)
        .ok_or_else(|| "IDX image file: data length overflows usize.".to_owned())?;

    if image_bytes.len() < required_image_len {
        return Err(format!(
            "IDX image file too short: header declares {} items of {}×{} pixels \
             ({} data bytes needed after header), but file is only {} bytes total.",
            n_items, rows, cols, n_items * n_pixels,
            image_bytes.len()
        ));
    }

    // ── Label file validation ───────────────────────────────────────────────

    if label_bytes.len() < 8 {
        return Err(format!(
            "IDX label file too short: expected at least 8 header bytes, got {}.",
            label_bytes.len()
        ));
    }

    if label_bytes[0] != 0x00 || label_bytes[1] != 0x00 {
        return Err(format!(
            "IDX label file: bytes 0-1 must be 0x00 0x00 (reserved), got 0x{:02X} 0x{:02X}.",
            label_bytes[0], label_bytes[1]
        ));
    }
    if label_bytes[2] != 0x08 {
        return Err(format!(
            "IDX label file: byte 2 (dtype) must be 0x08 (uint8), got 0x{:02X}.",
            label_bytes[2]
        ));
    }
    if label_bytes[3] != 0x01 {
        return Err(format!(
            "IDX label file: byte 3 (dimensions) must be 1, got {}. \
             This does not appear to be an IDX1 label file.",
            label_bytes[3]
        ));
    }

    let label_count = u32::from_be_bytes([
        label_bytes[4], label_bytes[5], label_bytes[6], label_bytes[7],
    ]) as usize;

    if label_count != n_items {
        return Err(format!(
            "IDX file mismatch: image file declares {} items but label file declares {}.",
            n_items, label_count
        ));
    }

    let required_label_len = 8 + n_items;
    if label_bytes.len() < required_label_len {
        return Err(format!(
            "IDX label file too short: header declares {} labels but file is only {} bytes \
             (need at least {} bytes).",
            n_items, label_bytes.len(), required_label_len
        ));
    }

    if n_classes < 2 {
        return Err(format!(
            "n_classes must be at least 2, got {}.",
            n_classes
        ));
    }

    // ── Build inputs ────────────────────────────────────────────────────────

    let image_data = &image_bytes[16..16 + n_items * n_pixels];
    let inputs: Vec<Vec<f64>> = image_data
        .chunks_exact(n_pixels)
        .map(|chunk| {
            chunk.iter().map(|&px| px as f64 / 255.0).collect()
        })
        .collect();

    // ── Build labels (one-hot) ───────────────────────────────────────────────

    let label_data = &label_bytes[8..8 + n_items];
    let mut labels: Vec<Vec<f64>> = Vec::with_capacity(n_items);
    for (i, &class_idx) in label_data.iter().enumerate() {
        let class = class_idx as usize;
        if class >= n_classes {
            return Err(format!(
                "IDX label at index {}: class index {} is out of range for n_classes={}.",
                i, class, n_classes
            ));
        }
        let mut one_hot = vec![0.0f64; n_classes];
        one_hot[class] = 1.0;
        labels.push(one_hot);
    }

    Ok((inputs, labels))
}
