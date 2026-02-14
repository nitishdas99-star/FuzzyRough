use approx::assert_abs_diff_eq;
use frlearn_core::{Estimator, Predictor};
use frlearn_neighbor::Metric;
use frlearn_neighbours::NN;
use ndarray::array;

#[test]
fn nn_predicts_obvious_nearest_class() {
    let x_train = array![[0.0, 0.0], [0.2, 0.1], [5.0, 5.0], [5.1, 4.9]];
    let y_train = vec![0usize, 0, 1, 1];
    let x_query = array![[0.1, 0.0], [5.0, 5.1]];

    let estimator = NN {
        k: 1,
        metric: Metric::Euclidean,
    };
    let model = estimator
        .fit(&x_train, &y_train)
        .expect("fit should succeed");
    let scores = model
        .predict_scores(&x_query)
        .expect("predict_scores should succeed");

    assert_eq!(scores.shape(), &[2, 2]);
    assert!(scores[[0, 0]] > scores[[0, 1]]);
    assert!(scores[[1, 1]] > scores[[1, 0]]);
    assert_abs_diff_eq!(scores.row(0).sum(), 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(scores.row(1).sum(), 1.0, epsilon = 1e-12);
}

#[test]
fn nn_score_shape_matches_query_and_classes() {
    let x_train = array![[0.0, 0.0], [1.0, 1.0], [9.0, 9.0]];
    let y_train = vec![0usize, 1, 2];
    let x_query = array![[0.2, 0.2], [1.1, 1.2], [8.9, 9.1], [2.0, 2.0]];

    let estimator = NN {
        k: 2,
        metric: Metric::Euclidean,
    };
    let model = estimator
        .fit(&x_train, &y_train)
        .expect("fit should succeed");
    let scores = model
        .predict_scores(&x_query)
        .expect("predict_scores should succeed");

    assert_eq!(scores.shape(), &[4, 3]);
}
