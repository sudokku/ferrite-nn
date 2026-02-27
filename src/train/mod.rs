pub mod trainer;
pub mod epoch_stats;
pub mod train_config;
pub mod loop_fn;

pub use trainer::train_network;
pub use epoch_stats::EpochStats;
pub use train_config::TrainConfig;
pub use loop_fn::train_loop;
