use approx::assert_abs_diff_eq;
use frlearn_core::{Estimator, Predictor};
use frlearn_neighbor::Metric;
use frlearn_neighbours::FRNN;
use ndarray::array;

#[test]
fn frnn_scores_have_expected_shape_and_no_nans() {
    let x_train = array![[0.0, 0.0], [0.2, 0.1], [3.0, 3.0], [3.1, 2.9], [6.0, 6.0]];
    let y_train = vec![0usize, 0, 1, 1, 2];
    let x_query = array![[0.1, 0.1], [3.0, 3.2], [5.9, 6.2]];

    let estimator = FRNN {
        k: 3,
        metric: Metric::Euclidean,
    };
    let model = estimator
        .fit(&x_train, &y_train)
        .expect("fit should succeed");
    let scores = model
        .predict_scores(&x_query)
        .expect("predict_scores should succeed");

    assert_eq!(scores.shape(), &[3, 3]);
    assert!(scores.iter().all(|value| !value.is_nan()));
    for row in scores.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-12);
    }
}

#[test]
fn frnn_is_deterministic_for_fixed_data() {
    let x_train = array![[0.0, 0.0], [1.0, 1.0], [1.2, 1.1], [4.0, 4.0]];
    let y_train = vec![0usize, 1, 1, 2];
    let x_query = array![[0.5, 0.5], [3.9, 3.8]];

    let estimator = FRNN {
        k: 3,
        metric: Metric::Euclidean,
    };
    let model = estimator
        .fit(&x_train, &y_train)
        .expect("fit should succeed");

    let scores_a = model
        .predict_scores(&x_query)
        .expect("predict_scores should succeed");
    let scores_b = model
        .predict_scores(&x_query)
        .expect("predict_scores should succeed");

    assert_eq!(scores_a.shape(), scores_b.shape());
    for row_idx in 0..scores_a.nrows() {
        for col_idx in 0..scores_a.ncols() {
            assert_abs_diff_eq!(
                scores_a[[row_idx, col_idx]],
                scores_b[[row_idx, col_idx]],
                epsilon = 1e-12
            );
        }
    }
}
