/// Image preprocessing utilities for the ferrite-nn studio.
///
/// These functions decode image bytes (PNG/JPEG/BMP/GIF), resize them to the
/// specified dimensions, and normalize pixel values to the [0, 1] range ready
/// for network inference.

/// Decodes image bytes, resizes to `width × height`, converts to grayscale,
/// and normalizes pixels to [0, 1].
///
/// Returns a flat `Vec<f64>` of length `width * height`.
pub fn image_bytes_to_grayscale_input(
    bytes: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<f64>, String> {
    let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
    let resized = img.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
    let gray = resized.to_luma8();
    Ok(gray.pixels().map(|p| p.0[0] as f64 / 255.0).collect())
}

/// Decodes image bytes, resizes to `width × height`, and flattens as R, G, B, ...
/// normalized to [0, 1].
///
/// Returns a flat `Vec<f64>` of length `width * height * 3`.
pub fn image_bytes_to_rgb_input(
    bytes: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<f64>, String> {
    let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
    let resized = img.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();
    Ok(rgb.pixels().flat_map(|p| p.0.iter().map(|&c| c as f64 / 255.0)).collect())
}
