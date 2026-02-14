//! Preprocessing models used by fuzzy-rough learners.
#![cfg_attr(doc, warn(missing_docs))]
#![warn(rust_2018_idioms)]
#![warn(clippy::all)]

pub mod range_normaliser;

use frlearn_core::{FrError, FrResult, Matrix};

pub use range_normaliser::{RangeNormaliser, RangeNormaliserModel};

pub trait TransformerModel {
    fn transform(&self, x: &Matrix) -> FrResult<Matrix>;
}

pub fn validate_feature_count(x: &Matrix, expected_features: usize) -> FrResult<()> {
    if x.ncols() != expected_features {
        return Err(FrError::InvalidInput(format!(
            "feature mismatch: expected {}, found {}",
            expected_features,
            x.ncols()
        )));
    }

    Ok(())
}
