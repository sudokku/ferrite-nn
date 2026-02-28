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
pub use network::metadata::{ModelMetadata, InputType};
pub use network::spec::{NetworkSpec, LayerSpec};
pub use loss::mse::MseLoss;
pub use loss::cross_entropy::CrossEntropyLoss;
pub use loss::bce::BceLoss;
pub use loss::mae::MaeLoss;
pub use loss::huber::HuberLoss;
pub use loss::loss_type::LossType;
pub use optim::sgd::Sgd;
pub use train::trainer::train_network;
pub use train::epoch_stats::EpochStats;
pub use train::train_config::TrainConfig;
pub use train::loop_fn::train_loop;
