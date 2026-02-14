//! Shared math primitives for fuzzy-rough algorithms.
#![cfg_attr(doc, warn(missing_docs))]
#![warn(rust_2018_idioms)]
#![warn(clippy::all)]

pub mod t_norms;
pub mod transformations;
pub mod vector_measures;
pub mod weights;

use thiserror::Error;

pub use t_norms::{
    lukasiewicz_t_norm, lukasiewicz_t_norm_scalar, min_t_norm, min_t_norm_scalar, product_t_norm,
    product_t_norm_scalar,
};
pub use transformations::{clamp01, complement, row_sums, safe_divide, safe_normalize_rows};
pub use vector_measures::{l1_norm, l2_distance, l2_norm, max_norm};
pub use weights::{decreasing_weights, uniform_weights};

#[derive(Debug, Error, PartialEq)]
pub enum MathError {
    #[error("vector length mismatch: left={left}, right={right}")]
    VectorLengthMismatch { left: usize, right: usize },
    #[error("shape mismatch: left=({left_rows}, {left_cols}), right=({right_rows}, {right_cols})")]
    ShapeMismatch {
        left_rows: usize,
        left_cols: usize,
        right_rows: usize,
        right_cols: usize,
    },
}

pub type MathResult<T> = Result<T, MathError>;
