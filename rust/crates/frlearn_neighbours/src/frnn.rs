use frlearn_core::{Estimator, FrError, FrResult, Labels, Matrix, Predictor};
use frlearn_math::{clamp01, complement, min_t_norm_scalar, safe_normalize_rows};
use frlearn_neighbor::{BruteForceIndex, Metric, NeighborIndex};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct FRNN {
    pub k: usize,
    pub metric: Metric,
}

impl Default for FRNN {
    fn default() -> Self {
        Self {
            k: 5,
            metric: Metric::Euclidean,
        }
    }
}

impl Estimator for FRNN {
    type Model = FRNNModel;

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

        Ok(FRNNModel {
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
pub struct FRNNModel {
    pub x_train: Matrix,
    pub y_train: Labels,
    pub index: BruteForceIndex,
    pub n_classes: usize,
    pub k: usize,
}

impl Predictor for FRNNModel {
    fn predict_scores(&self, xq: &Matrix) -> FrResult<Matrix> {
        let (indices, distances) = self.index.query(xq, self.k)?;
        let n_query = indices.nrows();
        let k_eff = indices.ncols();
        let mut scores = Matrix::zeros((n_query, self.n_classes));

        if k_eff == 0 {
            return Ok(scores);
        }

        for query_idx in 0..n_query {
            for class_idx in 0..self.n_classes {
                let mut upper_approximation: f64 = 0.0;
                let mut lower_approximation: f64 = 1.0;

                for neighbor_idx in 0..k_eff {
                    let train_idx = indices[[query_idx, neighbor_idx]];
                    let label = self.y_train[train_idx];
                    let similarity = similarity_from_distance(distances[[query_idx, neighbor_idx]]);

                    if label == class_idx {
                        upper_approximation = upper_approximation.max(similarity);
                    } else {
                        lower_approximation =
                            min_t_norm_scalar(lower_approximation, complement(similarity));
                    }
                }

                let score = 0.5 * (lower_approximation + upper_approximation);
                scores[[query_idx, class_idx]] = clamp01(score);
            }
        }

        Ok(safe_normalize_rows(&scores))
    }
}

fn similarity_from_distance(distance: f64) -> f64 {
    if !distance.is_finite() || distance < 0.0 {
        return 0.0;
    }

    clamp01(1.0 / (1.0 + distance))
}
