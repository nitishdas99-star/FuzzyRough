use ndarray::ArrayView1;

use crate::{MathError, MathResult};

pub fn l1_norm(row: ArrayView1<'_, f64>) -> f64 {
    row.iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(f64::abs)
        .sum()
}

pub fn l2_norm(row: ArrayView1<'_, f64>) -> f64 {
    row.iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

pub fn max_norm(row: ArrayView1<'_, f64>) -> f64 {
    row.iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(f64::abs)
        .fold(0.0, f64::max)
}

pub fn l2_distance(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> MathResult<f64> {
    if left.len() != right.len() {
        return Err(MathError::VectorLengthMismatch {
            left: left.len(),
            right: right.len(),
        });
    }

    let squared_sum = left
        .iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| {
            let lhs = if left_value.is_finite() {
                *left_value
            } else {
                0.0
            };
            let rhs = if right_value.is_finite() {
                *right_value
            } else {
                0.0
            };
            let delta = lhs - rhs;
            delta * delta
        })
        .sum::<f64>();

    Ok(squared_sum.sqrt())
}
