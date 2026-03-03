pub mod mse;
pub mod cross_entropy;
pub mod bce;
pub mod mae;
pub mod huber;
pub mod loss_type;

pub use mse::MseLoss;
pub use cross_entropy::CrossEntropyLoss;
pub use bce::BceLoss;
pub use mae::MaeLoss;
pub use huber::HuberLoss;
pub use loss_type::LossType;
