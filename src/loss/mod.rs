pub mod mse;
pub mod cross_entropy;
pub mod loss_type;

pub use mse::MseLoss;
pub use cross_entropy::CrossEntropyLoss;
pub use loss_type::LossType;
