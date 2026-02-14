use approx::assert_abs_diff_eq;
use frlearn_core::{
    Estimator, FrError, FrResult, Labels, Matrix, Predictor, probabilities_from_scores,
    select_class,
};
use ndarray::array;

#[derive(Debug, Default)]
struct DummyEstimator;

#[derive(Debug, Clone)]
struct DummyModel {
    class_count: usize,
}

impl Estimator for DummyEstimator {
    type Model = DummyModel;

    fn fit(&self, x: &Matrix, y: &Labels) -> FrResult<Self::Model> {
        if x.nrows() == 0 {
            return Err(FrError::EmptyInput);
        }
        if x.nrows() != y.len() {
            return Err(FrError::LabelLengthMismatch {
                expected: x.nrows(),
                found: y.len(),
            });
        }

        let class_count = y.iter().copied().max().unwrap_or(0) + 1;
        Ok(DummyModel { class_count })
    }
}

impl Predictor for DummyModel {
    fn predict_scores(&self, x: &Matrix) -> FrResult<Matrix> {
        if x.nrows() == 0 {
            return Err(FrError::EmptyInput);
        }
        if self.class_count == 0 {
            return Err(FrError::NotFitted);
        }

        let mut scores = Matrix::zeros((x.nrows(), self.class_count));
        for row_idx in 0..x.nrows() {
            let class_idx = row_idx % self.class_count;
            scores[[row_idx, class_idx]] = 1.0;
        }
        Ok(scores)
    }
}

#[test]
fn select_class_returns_argmax_indices() {
    let scores = array![[0.1, 0.9, 0.8], [2.0, 1.0, 2.0], [-1.0, -0.5, -3.0]];
    let labels = select_class(&scores);
    assert_eq!(labels, vec![1, 0, 1]);
}

#[test]
fn probabilities_rows_sum_to_one_and_contain_no_nans() {
    let scores = array![
        [2.0, 1.0, 1.0],                     // non-negative normalization path
        [-2.0, -1.0, -3.0],                  // softmax path
        [f64::NAN, f64::INFINITY, f64::NAN]  // all invalid values -> uniform
    ];
    let probabilities = probabilities_from_scores(&scores);

    for row in probabilities.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-12);
        assert!(row.iter().all(|value| !value.is_nan()));
    }
}

#[test]
fn probabilities_handles_all_zero_rows_with_uniform_distribution() {
    let scores = Matrix::zeros((2, 4));
    let probabilities = probabilities_from_scores(&scores);

    for row in probabilities.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-12);
        for value in row {
            assert_abs_diff_eq!(*value, 0.25, epsilon = 1e-12);
        }
    }
}

#[test]
fn estimator_predictor_contract_runs_end_to_end() {
    let x_train = Matrix::zeros((4, 2));
    let y_train = vec![0usize, 1, 0, 1];

    let estimator = DummyEstimator;
    let model = estimator
        .fit(&x_train, &y_train)
        .expect("fit should succeed");

    let x_query = Matrix::zeros((3, 2));
    let scores = model
        .predict_scores(&x_query)
        .expect("predict_scores should succeed");
    let labels = select_class(&scores);
    let probabilities = probabilities_from_scores(&scores);

    assert_eq!(scores.shape(), &[3, 2]);
    assert_eq!(labels, vec![0, 1, 0]);
    for row in probabilities.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-12);
    }
}
