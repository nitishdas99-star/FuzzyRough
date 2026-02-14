//! Core protocol types and utilities for fuzzy-rough models.
#![cfg_attr(doc, warn(missing_docs))]
#![warn(rust_2018_idioms)]
#![warn(clippy::all)]

pub mod error;
pub mod types;

use ndarray::Axis;

pub use error::FrError;
pub use types::{Labels, Matrix};

pub type FrResult<T> = Result<T, FrError>;

pub trait Estimator {
    type Model: Predictor;

    fn fit(&self, x: &Matrix, y: &Labels) -> FrResult<Self::Model>;
}

pub trait Predictor {
    fn predict_scores(&self, x: &Matrix) -> FrResult<Matrix>;
}

pub fn select_class(scores: &Matrix) -> Labels {
    if scores.ncols() == 0 {
        return vec![0; scores.nrows()];
    }

    scores
        .axis_iter(Axis(0))
        .map(|row| {
            let mut best_idx = 0usize;
            let mut best_value = f64::NEG_INFINITY;

            for (idx, value) in row.iter().copied().enumerate() {
                let comparable_value = if value.is_finite() {
                    value
                } else {
                    f64::NEG_INFINITY
                };

                if comparable_value > best_value {
                    best_value = comparable_value;
                    best_idx = idx;
                }
            }

            best_idx
        })
        .collect()
}

pub fn probabilities_from_scores(scores: &Matrix) -> Matrix {
    let n_rows = scores.nrows();
    let n_cols = scores.ncols();
    let mut probabilities = Matrix::zeros((n_rows, n_cols));

    if n_cols == 0 {
        return probabilities;
    }

    for (row_idx, row) in scores.axis_iter(Axis(0)).enumerate() {
        let mut use_simple_normalization = true;
        let mut simple_sum = 0.0f64;

        for value in row.iter().copied() {
            if !value.is_finite() || value < 0.0 {
                use_simple_normalization = false;
                break;
            }

            simple_sum += value;
            if !simple_sum.is_finite() {
                use_simple_normalization = false;
                break;
            }
        }

        if use_simple_normalization {
            if simple_sum > 0.0 {
                for (col_idx, value) in row.iter().copied().enumerate() {
                    probabilities[[row_idx, col_idx]] = value / simple_sum;
                }
            } else {
                fill_uniform_row(&mut probabilities, row_idx, n_cols);
            }
            continue;
        }

        let row_max = row
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        if !row_max.is_finite() {
            fill_uniform_row(&mut probabilities, row_idx, n_cols);
            continue;
        }

        let mut softmax_sum = 0.0f64;
        for (col_idx, value) in row.iter().copied().enumerate() {
            let weight = if value.is_finite() {
                (value - row_max).exp()
            } else {
                0.0
            };
            probabilities[[row_idx, col_idx]] = weight;
            softmax_sum += weight;
        }

        if softmax_sum > 0.0 && softmax_sum.is_finite() {
            for col_idx in 0..n_cols {
                probabilities[[row_idx, col_idx]] /= softmax_sum;
            }
        } else {
            fill_uniform_row(&mut probabilities, row_idx, n_cols);
        }
    }

    probabilities
}

fn fill_uniform_row(probabilities: &mut Matrix, row_idx: usize, n_cols: usize) {
    let uniform_probability = 1.0 / n_cols as f64;
    for col_idx in 0..n_cols {
        probabilities[[row_idx, col_idx]] = uniform_probability;
    }
}
