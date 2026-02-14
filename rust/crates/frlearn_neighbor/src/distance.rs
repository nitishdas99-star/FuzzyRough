use frlearn_core::{FrError, FrResult};
use ndarray::ArrayView1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    Manhattan,
}

pub fn pairwise_distance(
    left: ArrayView1<'_, f64>,
    right: ArrayView1<'_, f64>,
    metric: Metric,
) -> FrResult<f64> {
    if left.len() != right.len() {
        return Err(FrError::InvalidInput(format!(
            "distance feature mismatch: left={}, right={}",
            left.len(),
            right.len()
        )));
    }

    match metric {
        Metric::Euclidean => euclidean_distance(left, right),
        Metric::Manhattan => manhattan_distance(left, right),
    }
}

fn euclidean_distance(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> FrResult<f64> {
    let mut squared_sum = 0.0;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        let lhs = validate_finite(*left_value)?;
        let rhs = validate_finite(*right_value)?;
        let delta = lhs - rhs;
        squared_sum += delta * delta;
    }

    Ok(squared_sum.sqrt())
}

fn manhattan_distance(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> FrResult<f64> {
    let mut sum = 0.0;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        let lhs = validate_finite(*left_value)?;
        let rhs = validate_finite(*right_value)?;
        sum += (lhs - rhs).abs();
    }

    Ok(sum)
}

fn validate_finite(value: f64) -> FrResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FrError::InvalidInput(
            "non-finite value encountered".to_string(),
        ))
    }
}
