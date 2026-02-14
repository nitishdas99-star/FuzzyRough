use frlearn_core::{Estimator, FrError, FrResult, Labels, Matrix, Predictor};
use frlearn_neighbor::{BruteForceIndex, Metric, NeighborIndex};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct NN {
    pub k: usize,
    pub metric: Metric,
}

impl Default for NN {
    fn default() -> Self {
        Self {
            k: 1,
            metric: Metric::Euclidean,
        }
    }
}

impl Estimator for NN {
    type Model = NNModel;

    fn fit(&self, x: &Matrix, y: &Labels) -> FrResult<Self::Model> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(FrError::EmptyInput);
        }
        if self.k == 0 {
            return Err(FrError::InvalidInput("k must be at least 1".to_string()));
        }
        if y.len() != x.nrows() {
            return Err(FrError::LabelLengthMismatch {
                expected: x.nrows(),
                found: y.len(),
            });
        }

        let n_classes = y
            .iter()
            .copied()
            .max()
            .map(|max_label| max_label + 1)
            .ok_or_else(|| FrError::InvalidInput("labels cannot be empty".to_string()))?;

        Ok(NNModel {
            x_train: x.clone(),
            y_train: y.clone(),
            index: BruteForceIndex::new(x.clone(), self.metric),
            n_classes,
            k: self.k,
        })
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct NNModel {
    pub x_train: Matrix,
    pub y_train: Labels,
    pub index: BruteForceIndex,
    pub n_classes: usize,
    pub k: usize,
}

impl Predictor for NNModel {
    fn predict_scores(&self, xq: &Matrix) -> FrResult<Matrix> {
        let (indices, distances) = self.index.query(xq, self.k)?;
        let n_query = indices.nrows();
        let k_eff = indices.ncols();
        let mut scores = Matrix::zeros((n_query, self.n_classes));

        if k_eff == 0 {
            return Ok(scores);
        }

        for query_idx in 0..n_query {
            let mut total_weight = 0.0;

            for neighbor_idx in 0..k_eff {
                let train_idx = indices[[query_idx, neighbor_idx]];
                let class_label = self.y_train[train_idx];
                let distance = distances[[query_idx, neighbor_idx]];
                let weight = if distance <= 0.0 {
                    1.0
                } else {
                    1.0 / (1.0 + distance)
                };
                scores[[query_idx, class_label]] += weight;
                total_weight += weight;
            }

            if total_weight > 0.0 && total_weight.is_finite() {
                for class_idx in 0..self.n_classes {
                    scores[[query_idx, class_idx]] /= total_weight;
                }
            }
        }

        Ok(scores)
    }
}
