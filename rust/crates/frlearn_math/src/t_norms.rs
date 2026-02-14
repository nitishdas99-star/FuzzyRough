use ndarray::Array2;

use crate::transformations::clamp01;
use crate::{MathError, MathResult};

pub fn min_t_norm_scalar(a: f64, b: f64) -> f64 {
    clamp01(clamp01(a).min(clamp01(b)))
}

pub fn product_t_norm_scalar(a: f64, b: f64) -> f64 {
    clamp01(clamp01(a) * clamp01(b))
}

pub fn lukasiewicz_t_norm_scalar(a: f64, b: f64) -> f64 {
    clamp01((clamp01(a) + clamp01(b) - 1.0).max(0.0))
}

pub fn min_t_norm(left: &Array2<f64>, right: &Array2<f64>) -> MathResult<Array2<f64>> {
    elementwise_t_norm(left, right, min_t_norm_scalar)
}

pub fn product_t_norm(left: &Array2<f64>, right: &Array2<f64>) -> MathResult<Array2<f64>> {
    elementwise_t_norm(left, right, product_t_norm_scalar)
}

pub fn lukasiewicz_t_norm(left: &Array2<f64>, right: &Array2<f64>) -> MathResult<Array2<f64>> {
    elementwise_t_norm(left, right, lukasiewicz_t_norm_scalar)
}

fn elementwise_t_norm<F>(
    left: &Array2<f64>,
    right: &Array2<f64>,
    t_norm_fn: F,
) -> MathResult<Array2<f64>>
where
    F: Fn(f64, f64) -> f64,
{
    if left.dim() != right.dim() {
        return Err(MathError::ShapeMismatch {
            left_rows: left.nrows(),
            left_cols: left.ncols(),
            right_rows: right.nrows(),
            right_cols: right.ncols(),
        });
    }

    let mut result = Array2::<f64>::zeros(left.raw_dim());
    for row_idx in 0..left.nrows() {
        for col_idx in 0..left.ncols() {
            result[[row_idx, col_idx]] =
                t_norm_fn(left[[row_idx, col_idx]], right[[row_idx, col_idx]]);
        }
    }

    Ok(result)
}
