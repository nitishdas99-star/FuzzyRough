use frlearn_core::{FrResult, Matrix};
use ndarray::Array2;

use crate::distance::Metric;
use crate::{NeighborIndex, pairwise_distance, validate_query_features};

#[derive(Debug, Clone)]
pub struct BruteForceIndex {
    pub x_train: Matrix,
    pub metric: Metric,
}

impl BruteForceIndex {
    pub fn new(x_train: Matrix, metric: Metric) -> Self {
        Self { x_train, metric }
    }
}

impl NeighborIndex for BruteForceIndex {
    fn query(&self, xq: &Matrix, k: usize) -> FrResult<(Array2<usize>, Matrix)> {
        validate_query_features(&self.x_train, xq)?;

        let n_query = xq.nrows();
        let n_train = self.x_train.nrows();
        let effective_k = k.min(n_train);

        let mut indices = Array2::<usize>::zeros((n_query, effective_k));
        let mut distances = Matrix::zeros((n_query, effective_k));

        if n_query == 0 || effective_k == 0 {
            return Ok((indices, distances));
        }

        for (query_idx, query_row) in xq.outer_iter().enumerate() {
            let mut row_distances = self
                .x_train
                .outer_iter()
                .enumerate()
                .map(|(train_idx, train_row)| {
                    let distance = pairwise_distance(query_row, train_row, self.metric)?;
                    Ok((train_idx, distance))
                })
                .collect::<FrResult<Vec<_>>>()?;

            row_distances.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });

            for neighbor_idx in 0..effective_k {
                let (train_idx, distance) = row_distances[neighbor_idx];
                indices[[query_idx, neighbor_idx]] = train_idx;
                distances[[query_idx, neighbor_idx]] = distance;
            }
        }

        Ok((indices, distances))
    }
}
