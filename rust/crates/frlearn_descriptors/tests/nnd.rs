use approx::assert_abs_diff_eq;
use frlearn_core::Predictor;
use frlearn_descriptors::NND;
use frlearn_neighbor::Metric;
use ndarray::array;

#[test]
fn inliers_score_lower_than_far_outliers() {
    let x_inlier = array![[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1], [0.05, 0.05]];
    let x_query = array![[0.0, 0.0], [4.0, 4.0], [0.2, -0.1], [5.0, 5.0]];

    let descriptor = NND {
        k: 2,
        metric: Metric::Euclidean,
    };
    let model = descriptor.fit(&x_inlier).expect("fit should succeed");
    let scores = model
        .predict_anomaly_scores(&x_query)
        .expect("predict should succeed");

    assert!(scores[0] < scores[1]);
    assert!(scores[2] < scores[3]);
}

#[test]
fn scores_are_deterministic_and_finite() {
    let x_inlier = array![[1.0, 1.0], [1.2, 1.1], [0.8, 0.9], [1.1, 1.2]];
    let x_query = array![[1.0, 1.0], [1.5, 1.5], [2.0, 2.0]];

    let descriptor = NND {
        k: 3,
        metric: Metric::Euclidean,
    };
    let model = descriptor.fit(&x_inlier).expect("fit should succeed");

    let first = model
        .predict_anomaly_scores(&x_query)
        .expect("predict should succeed");
    let second = model
        .predict_anomaly_scores(&x_query)
        .expect("predict should succeed");

    assert_eq!(first.len(), second.len());
    for idx in 0..first.len() {
        assert!(!first[idx].is_nan());
        assert_abs_diff_eq!(first[idx], second[idx], epsilon = 1e-12);
    }
}

#[test]
fn predictor_interface_returns_single_score_column() {
    let x_inlier = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let x_query = array![[0.1, 0.1], [1.5, 1.5]];

    let descriptor = NND {
        k: 2,
        metric: Metric::Euclidean,
    };
    let model = descriptor.fit(&x_inlier).expect("fit should succeed");
    let score_matrix = model
        .predict_scores(&x_query)
        .expect("predict should succeed");

    assert_eq!(score_matrix.shape(), &[2, 1]);
    for value in score_matrix.iter() {
        assert!(!value.is_nan());
        assert!((0.0..=1.0).contains(value));
    }
}
