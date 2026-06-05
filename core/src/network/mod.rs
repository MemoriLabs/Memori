pub mod client;
pub mod error;

pub use client::{ApiSubdomain, MemoriClient, resolve_base_url, resolve_x_api_key};
pub use error::ApiError;
