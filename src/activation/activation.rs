use serde::{Serialize, Deserialize};
use std::f64::consts::E;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Identity,
    /// Softmax is a vector-valued activation; it is applied at the layer level
    /// (not element-wise) in `Layer::feed_from()`.  The element-wise `function()`
    /// and `derivative()` methods are therefore not used for this variant.
    Softmax,
}

impl ActivationFunction {
    /// Element-wise activation.  For `Softmax`, call `Layer::feed_from()` which
    /// applies the full-vector softmax; this path should not be reached.
    pub fn function(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + E.powf(-x)),
            ActivationFunction::ReLU => if x > 0.0 { x } else { 0.0 },
            ActivationFunction::Identity => x,
            ActivationFunction::Softmax => {
                // Softmax cannot be applied element-wise; the layer handles it.
                panic!("ActivationFunction::Softmax::function() must not be called directly; \
                        use Layer::feed_from() which applies the full-vector softmax.")
            }
        }
    }

    /// Element-wise derivative of the activation.
    ///
    /// For `Softmax`, the layer pairs it with cross-entropy and the combined
    /// gradient is `predicted - expected` (already computed by
    /// `CrossEntropyLoss::derivative()`).  Returning `1.0` here lets
    /// `compute_gradients()` pass that delta through unchanged without
    /// double-applying the Jacobian.
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => {
                let fx = self.function(x);
                fx * (1.0 - fx)
            },
            ActivationFunction::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFunction::Identity => 1.0,
            ActivationFunction::Softmax => 1.0,
        }
    }
}
