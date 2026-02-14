use frlearn_core::{FrError, FrResult, Matrix};
use frlearn_math::{clamp01, safe_divide};
use ndarray::Array1;

use crate::{TransformerModel, validate_feature_count};

#[derive(Debug, Clone, Copy)]
pub struct RangeNormaliser {
    pub eps: f64,
}

impl Default for RangeNormaliser {
    fn default() -> Self {
        Self { eps: 1e-12 }
    }
}

impl RangeNormaliser {
    pub fn fit(&self, x: &Matrix) -> RangeNormaliserModel {
        let n_cols = x.ncols();
        let mut min = Array1::<f64>::zeros(n_cols);
        let mut max = Array1::<f64>::zeros(n_cols);

        if x.nrows() > 0 {
            for col_idx in 0..n_cols {
                let column = x.column(col_idx);
                let min_value = column
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .fold(f64::INFINITY, f64::min);
                let max_value = column
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .fold(f64::NEG_INFINITY, f64::max);

                min[col_idx] = if min_value.is_finite() {
                    min_value
                } else {
                    0.0
                };
                max[col_idx] = if max_value.is_finite() {
                    max_value
                } else {
                    0.0
                };
            }
        }

        RangeNormaliserModel {
            min,
            max,
            eps: self.eps.max(0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RangeNormaliserModel {
    pub min: Array1<f64>,
    pub max: Array1<f64>,
    pub eps: f64,
}

impl RangeNormaliserModel {
    pub fn transform(&self, x: &Matrix) -> Matrix {
        self.try_transform(x)
            .unwrap_or_else(|_| Matrix::zeros((x.nrows(), x.ncols())))
    }

    fn try_transform(&self, x: &Matrix) -> FrResult<Matrix> {
        if self.min.len() != self.max.len() {
            return Err(FrError::InvalidInput(format!(
                "model min/max length mismatch: {} vs {}",
                self.min.len(),
                self.max.len()
            )));
        }

        validate_feature_count(x, self.min.len())?;

        let mut output = Matrix::zeros((x.nrows(), x.ncols()));
        for row_idx in 0..x.nrows() {
            for col_idx in 0..x.ncols() {
                let value = x[[row_idx, col_idx]];
                let numerator = if value.is_finite() {
                    value - self.min[col_idx]
                } else {
                    0.0
                };
                let denominator = self.max[col_idx] - self.min[col_idx] + self.eps;
                let scaled = safe_divide(numerator, denominator, 0.0);
                output[[row_idx, col_idx]] = clamp01(scaled);
            }
        }

        Ok(output)
    }
}

impl TransformerModel for RangeNormaliserModel {
    fn transform(&self, x: &Matrix) -> FrResult<Matrix> {
        self.try_transform(x)
    }
}
