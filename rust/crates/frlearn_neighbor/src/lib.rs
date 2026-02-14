//! Neighbour-index abstractions and distance computation backends.
#![cfg_attr(doc, warn(missing_docs))]
#![warn(rust_2018_idioms)]
#![warn(clippy::all)]

pub mod bruteforce;
pub mod distance;

use frlearn_core::{FrError, FrResult, Matrix};
use ndarray::Array2;

pub use bruteforce::BruteForceIndex;
pub use distance::{Metric, pairwise_distance};

pub trait NeighborIndex {
    fn query(&self, xq: &Matrix, k: usize) -> FrResult<(Array2<usize>, Matrix)>;
}

fn validate_query_features(x_train: &Matrix, xq: &Matrix) -> FrResult<()> {
    if xq.ncols() != x_train.ncols() {
        return Err(FrError::InvalidInput(format!(
            "query feature mismatch: expected {}, found {}",
            x_train.ncols(),
            xq.ncols()
        )));
    }

    Ok(())
}
