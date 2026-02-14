use std::fs;
use std::path::{Path, PathBuf};

use approx::assert_abs_diff_eq;
use frlearn_core::Matrix;
use frlearn_preprocess::RangeNormaliser;
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

#[test]
fn range_normaliser_matches_python_fixture() {
    let fixture = read_json("x_norm.json");
    let x_train = matrix_from_value(&fixture["x_train"]);
    let expected_norm = matrix_from_value(&fixture["x_train_norm"]);
    let eps = fixture["eps"].as_f64().expect("eps should be f64");

    let model = RangeNormaliser { eps }.fit(&x_train);
    let actual_norm = model.transform(&x_train);

    assert_eq!(actual_norm.shape(), expected_norm.shape());
    for row_idx in 0..actual_norm.nrows() {
        for col_idx in 0..actual_norm.ncols() {
            assert_abs_diff_eq!(
                actual_norm[[row_idx, col_idx]],
                expected_norm[[row_idx, col_idx]],
                epsilon = 1e-10
            );
        }
    }
}
