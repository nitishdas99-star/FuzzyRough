use frlearn_core::{FrError, FrResult, Matrix, Predictor};
use frlearn_math::safe_divide;
use frlearn_neighbor::{BruteForceIndex, Metric, NeighborIndex};
use ndarray::Array2;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct NND {
    pub k: usize,
    pub metric: Metric,
}

impl Default for NND {
    fn default() -> Self {
        Self {
            k: 5,
            metric: Metric::Euclidean,
        }
    }
}

impl NND {
    pub fn fit(&self, x_inlier: &Matrix) -> FrResult<NNDModel> {
        if x_inlier.nrows() == 0 || x_inlier.ncols() == 0 {
            return Err(FrError::EmptyInput);
        }
        if self.k == 0 {
            return Err(FrError::InvalidInput("k must be at least 1".to_string()));
        }

        Ok(NNDModel {
            x_inlier: x_inlier.clone(),
            index: BruteForceIndex::new(x_inlier.clone(), self.metric),
            k: self.k,
            metric: self.metric,
        })
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct NNDModel {
    pub x_inlier: Matrix,
    pub index: BruteForceIndex,
    pub k: usize,
    pub metric: Metric,
}

impl NNDModel {
    pub fn predict_anomaly_scores(&self, xq: &Matrix) -> FrResult<Vec<f64>> {
        let (_indices, distances) = self.index.query(xq, self.k)?;
        let k_eff = distances.ncols();

        if k_eff == 0 {
            return Ok(vec![0.0; xq.nrows()]);
        }

        let mut raw_scores = Vec::with_capacity(distances.nrows());
        for row in distances.rows() {
            let distance_sum = row.iter().copied().sum::<f64>();
            let avg_distance = safe_divide(distance_sum, k_eff as f64, 0.0);
            raw_scores.push(avg_distance.max(0.0));
        }

        Ok(normalize_scores(raw_scores))
    }
}

impl Predictor for NNDModel {
    fn predict_scores(&self, xq: &Matrix) -> FrResult<Matrix> {
        let anomaly_scores = self.predict_anomaly_scores(xq)?;
        let mut output = Array2::<f64>::zeros((anomaly_scores.len(), 1));
        for (row_idx, score) in anomaly_scores.into_iter().enumerate() {
            output[[row_idx, 0]] = score;
        }
        Ok(output)
    }
}

fn normalize_scores(raw_scores: Vec<f64>) -> Vec<f64> {
    if raw_scores.is_empty() {
        return raw_scores;
    }

    let min_score = raw_scores.iter().copied().fold(f64::INFINITY, f64::min);
    let max_score = raw_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_score - min_score;

    if !range.is_finite() || range <= f64::EPSILON {
        return vec![0.0; raw_scores.len()];
    }

    raw_scores
        .into_iter()
        .map(|score| safe_divide(score - min_score, range, 0.0).clamp(0.0, 1.0))
        .collect()
}
