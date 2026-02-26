pub mod math;
pub mod activation;
pub mod layers;
pub mod network;
pub mod loss;
pub mod optim;
pub mod train;

// Convenience re-exports
pub use math::matrix::Matrix;
pub use activation::activation::ActivationFunction;
pub use layers::dense::Layer;
pub use network::network::Network;
pub use loss::mse::MseLoss;
pub use optim::sgd::Sgd;
pub use train::trainer::train_network;
