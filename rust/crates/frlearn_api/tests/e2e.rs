use approx::assert_abs_diff_eq;
use frlearn_api::{Metric, default_nn_pipeline};
use ndarray::array;

#[test]
fn e2e_pipeline_predicts_with_expected_length() {
    let x_train = array![
        [0.0, 0.0],
        [0.2, 0.1],
        [4.9, 5.0],
        [5.1, 4.8],
        [9.9, 10.1],
        [10.2, 9.8]
    ];
    let y_train = vec![0usize, 0, 1, 1, 2, 2];
    let x_query = array![[0.1, 0.0], [5.0, 4.9], [10.0, 10.0], [4.8, 5.1]];

    let pipeline = default_nn_pipeline(3, Metric::Euclidean);
    let model = pipeline
        .fit(&x_train, &y_train)
        .expect("fit should succeed");

    let predictions = model.predict(&x_query).expect("predict should succeed");
    assert_eq!(predictions.len(), x_query.nrows());
}

#[test]
fn e2e_predict_proba_rows_sum_to_one() {
    let x_train = array![
        [1.0, 1.0],
        [1.2, 0.9],
        [5.0, 5.0],
        [4.8, 5.2],
        [8.8, 9.0],
        [9.1, 8.9]
    ];
    let y_train = vec![0usize, 0, 1, 1, 2, 2];
    let x_query = array![[1.1, 1.0], [5.1, 5.1], [9.0, 8.8]];

    let pipeline = default_nn_pipeline(3, Metric::Euclidean);
    let model = pipeline
        .fit(&x_train, &y_train)
        .expect("fit should succeed");
    let probabilities = model
        .predict_proba(&x_query)
        .expect("predict_proba should succeed");

    assert_eq!(probabilities.nrows(), x_query.nrows());
    assert_eq!(probabilities.ncols(), 3);
    for row in probabilities.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-12);
    }
}

#[test]
fn e2e_pipeline_runs_without_panics() {
    let x_train = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let y_train = vec![0usize, 1, 2];
    let x_query = array![[0.1, 0.1], [1.1, 1.1], [1.9, 2.1]];

    let run = std::panic::catch_unwind(|| {
        let pipeline = default_nn_pipeline(2, Metric::Euclidean);
        let model = pipeline
            .fit(&x_train, &y_train)
            .expect("fit should succeed");
        let _predictions = model.predict(&x_query).expect("predict should succeed");
        let _probas = model
            .predict_proba(&x_query)
            .expect("predict_proba should succeed");
    });

    assert!(run.is_ok());
}
