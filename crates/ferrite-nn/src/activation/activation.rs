use serde::{Serialize, Deserialize};
use std::f64::consts::{E, PI};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Identity,
    /// Softmax is a vector-valued activation; it is applied at the layer level
    /// (not element-wise) in `Layer::feed_from()`.  The element-wise `function()`
    /// and `derivative()` methods are therefore not used for this variant.
    Softmax,
    Tanh,
    LeakyReLU { alpha: f64 },
    Elu { alpha: f64 },
    Gelu,
    Swish,
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
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::LeakyReLU { alpha } => if x > 0.0 { x } else { alpha * x },
            ActivationFunction::Elu { alpha } => {
                if x > 0.0 { x } else { alpha * (E.powf(x) - 1.0) }
            }
            ActivationFunction::Gelu => {
                let c = (2.0_f64 / PI).sqrt();
                0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
            }
            ActivationFunction::Swish => x / (1.0 + E.powf(-x)),
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
            ActivationFunction::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationFunction::LeakyReLU { alpha } => if x > 0.0 { 1.0 } else { *alpha },
            ActivationFunction::Elu { alpha } => {
                if x > 0.0 { 1.0 } else { alpha * E.powf(x) }
            }
            ActivationFunction::Gelu => {
                let c = (2.0_f64 / PI).sqrt();
                let inner = c * (x + 0.044715 * x.powi(3));
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner * tanh_inner;
                let d_inner = c * (1.0 + 3.0 * 0.044715 * x.powi(2));
                0.5 * tanh_inner + 0.5 * x * sech2 * d_inner + 0.5
            }
            ActivationFunction::Swish => {
                let sig = 1.0 / (1.0 + E.powf(-x));
                sig + x * sig * (1.0 - sig)
            }
        }
    }
}
