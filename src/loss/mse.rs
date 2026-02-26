pub struct MseLoss;

impl MseLoss {
    /// Scalar MSE: mean((predicted - expected)²)
    pub fn loss(predicted: &[f64], expected: &[f64]) -> f64 {
        let n = predicted.len() as f64;
        predicted.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / n
    }

    /// Per-output gradient: predicted - expected
    pub fn derivative(predicted: &[f64], expected: &[f64]) -> Vec<f64> {
        predicted.iter().zip(expected.iter())
            .map(|(a, b)| a - b)
            .collect()
    }
}
