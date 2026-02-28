pub struct BceLoss;

const EPS: f64 = 1e-12;

impl BceLoss {
    /// Scalar BCE: -mean(y·log(p+ε) + (1-y)·log(1-p+ε))
    pub fn loss(predicted: &[f64], expected: &[f64]) -> f64 {
        let n = predicted.len() as f64;
        predicted.iter().zip(expected.iter())
            .map(|(p, y)| -(y * (p + EPS).ln() + (1.0 - y) * (1.0 - p + EPS).ln()))
            .sum::<f64>() / n
    }

    /// Per-output gradient: (p - y) / ((p + ε) · (1 - p + ε))
    pub fn derivative(predicted: &[f64], expected: &[f64]) -> Vec<f64> {
        predicted.iter().zip(expected.iter())
            .map(|(p, y)| (p - y) / ((p + EPS) * (1.0 - p + EPS)))
            .collect()
    }
}
