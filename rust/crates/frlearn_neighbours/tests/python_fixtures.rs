use std::fs;
use std::path::{Path, PathBuf};

use approx::assert_abs_diff_eq;
use frlearn_core::{Estimator, Labels, Matrix, Predictor};
use frlearn_neighbor::Metric;
use frlearn_neighbours::{FRNN, NN};
use ndarray::Array2;
use serde_json::Value;

fn fixture_path(file_name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("python_reference")
        .join(file_name)
}

fn read_json(file_name: &str) -> Value {
    let path = fixture_path(file_name);
    let raw = fs::read_to_string(path).expect("fixture file should exist");
    serde_json::from_str(&raw).expect("fixture JSON should be valid")
}

fn matrix_from_value(value: &Value) -> Matrix {
    let rows = value
        .as_array()
        .expect("matrix JSON should be a 2D array structure");
    let n_rows = rows.len();
    let n_cols = rows
        .first()
        .and_then(|row| row.as_array())
        .map_or(0, |row| row.len());
    let flat = rows
        .iter()
        .flat_map(|row| {
            row.as_array()
                .expect("matrix row should be array")
                .iter()
                .map(|v| v.as_f64().expect("matrix entry should be f64"))
        })
        .collect::<Vec<_>>();
    Array2::from_shape_vec((n_rows, n_cols), flat).expect("matrix shape should be consistent")
}

fn labels_from_value(value: &Value) -> Labels {
    value
        .as_array()
        .expect("labels should be array")
        .iter()
        .map(|v| v.as_u64().expect("label should be u64") as usize)
        .collect()
}

#[test]
fn nn_scores_match_python_fixture() {
    let x_norm_fixture = read_json("x_norm.json");
    let scores_fixture = read_json("scores.json");

    let x_train_norm = matrix_from_value(&x_norm_fixture["x_train_norm"]);
    let x_query_norm = matrix_from_value(&x_norm_fixture["x_query_norm"]);
    let y_train = labels_from_value(&scores_fixture["y_train"]);
    let expected = matrix_from_value(&scores_fixture["nn_scores"]);
    let k = scores_fixture["k"].as_u64().expect("k should be u64") as usize;

    assert_eq!(
        scores_fixture["metric"]
            .as_str()
            .expect("metric should be string"),
        "euclidean"
    );

    let model = NN {
        k,
        metric: Metric::Euclidean,
    }
    .fit(&x_train_norm, &y_train)
    .expect("fit should succeed");
    let actual = model
        .predict_scores(&x_query_norm)
        .expect("predict_scores should succeed");

    assert_eq!(actual.shape(), expected.shape());
    for row_idx in 0..actual.nrows() {
        for col_idx in 0..actual.ncols() {
            assert_abs_diff_eq!(
                actual[[row_idx, col_idx]],
                expected[[row_idx, col_idx]],
                epsilon = 1e-10
            );
        }
    }
}

#[test]
fn frnn_scores_match_python_fixture() {
    let x_norm_fixture = read_json("x_norm.json");
    let scores_fixture = read_json("scores.json");

    let x_train_norm = matrix_from_value(&x_norm_fixture["x_train_norm"]);
    let x_query_norm = matrix_from_value(&x_norm_fixture["x_query_norm"]);
    let y_train = labels_from_value(&scores_fixture["y_train"]);
    let expected = matrix_from_value(&scores_fixture["frnn_scores"]);
    let k = scores_fixture["k"].as_u64().expect("k should be u64") as usize;

    let model = FRNN {
        k,
        metric: Metric::Euclidean,
    }
    .fit(&x_train_norm, &y_train)
    .expect("fit should succeed");
    let actual = model
        .predict_scores(&x_query_norm)
        .expect("predict_scores should succeed");

    assert_eq!(actual.shape(), expected.shape());
    for row_idx in 0..actual.nrows() {
        for col_idx in 0..actual.ncols() {
            assert_abs_diff_eq!(
                actual[[row_idx, col_idx]],
                expected[[row_idx, col_idx]],
                epsilon = 1e-10
            );
        }
    }
}
