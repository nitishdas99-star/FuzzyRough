use ndarray::{Array1, Array2};

pub fn clamp01(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }

    value.clamp(0.0, 1.0)
}

pub fn complement(value: f64) -> f64 {
    clamp01(1.0 - clamp01(value))
}

pub fn safe_divide(numerator: f64, denominator: f64, default: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
        return default;
    }

    let ratio = numerator / denominator;
    if ratio.is_finite() { ratio } else { default }
}

pub fn row_sums(matrix: &Array2<f64>) -> Array1<f64> {
    matrix
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .copied()
                .filter(|value| value.is_finite())
                .sum::<f64>()
        })
        .collect()
}

pub fn safe_normalize_rows(matrix: &Array2<f64>) -> Array2<f64> {
    let n_rows = matrix.nrows();
    let n_cols = matrix.ncols();
    let mut normalized = Array2::<f64>::zeros((n_rows, n_cols));

    if n_cols == 0 {
        return normalized;
    }

    for row_idx in 0..n_rows {
        let mut sum = 0.0;
        for col_idx in 0..n_cols {
            let value = matrix[[row_idx, col_idx]];
            let sanitized = if value.is_finite() && value > 0.0 {
                value
            } else {
                0.0
            };
            normalized[[row_idx, col_idx]] = sanitized;
            sum += sanitized;
        }

        if sum > 0.0 && sum.is_finite() {
            for col_idx in 0..n_cols {
                normalized[[row_idx, col_idx]] /= sum;
            }
        } else {
            let uniform = 1.0 / n_cols as f64;
            for col_idx in 0..n_cols {
                normalized[[row_idx, col_idx]] = uniform;
            }
        }
    }

    normalized
}
