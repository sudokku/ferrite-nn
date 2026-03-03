pub struct HuberLoss;

// Fixed δ = 1.0 keeps the enum variant unit (no f64 field) → preserves Eq + Copy.
const DELTA: f64 = 1.0;

impl HuberLoss {
    /// Scalar Huber: mean(h(predicted − expected))
    /// where h(x) = 0.5·x²  if |x| ≤ δ
    ///              δ·(|x| − 0.5·δ)  otherwise
    pub fn loss(predicted: &[f64], expected: &[f64]) -> f64 {
        let n = predicted.len() as f64;
        predicted.iter().zip(expected.iter())
            .map(|(p, y)| {
                let x = p - y;
                if x.abs() <= DELTA {
                    0.5 * x * x
                } else {
                    DELTA * (x.abs() - 0.5 * DELTA)
                }
            })
            .sum::<f64>() / n
    }

    /// Per-output gradient: x  if |x| ≤ δ,  else δ·sign(x)
    pub fn derivative(predicted: &[f64], expected: &[f64]) -> Vec<f64> {
        predicted.iter().zip(expected.iter())
            .map(|(p, y)| {
                let x = p - y;
                if x.abs() <= DELTA { x } else { DELTA * x.signum() }
            })
            .collect()
    }
}
