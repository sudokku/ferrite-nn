pub struct MaeLoss;

impl MaeLoss {
    /// Scalar MAE: mean(|predicted - expected|)
    pub fn loss(predicted: &[f64], expected: &[f64]) -> f64 {
        let n = predicted.len() as f64;
        predicted.iter().zip(expected.iter())
            .map(|(p, y)| (p - y).abs())
            .sum::<f64>() / n
    }

    /// Per-output subgradient: sign(p - y) / n  (0 when equal)
    pub fn derivative(predicted: &[f64], expected: &[f64]) -> Vec<f64> {
        let n = predicted.len() as f64;
        predicted.iter().zip(expected.iter())
            .map(|(p, y)| {
                let diff = p - y;
                if diff > 0.0 { 1.0 / n } else if diff < 0.0 { -1.0 / n } else { 0.0 }
            })
            .collect()
    }
}
