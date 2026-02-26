use std::f64::consts::E;

#[derive(Debug, Clone)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Identity,
    // TODO: Add more variants (Tanh, Softmax, LeakyReLU, …)
}

impl ActivationFunction {
    pub fn function(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + E.powf(-x)),
            ActivationFunction::ReLU => if x > 0.0 { x } else { 0.0 },
            ActivationFunction::Identity => x,
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => {
                let fx = self.function(x);
                fx * (1.0 - fx)
            },
            ActivationFunction::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFunction::Identity => 1.0,
        }
    }
}
