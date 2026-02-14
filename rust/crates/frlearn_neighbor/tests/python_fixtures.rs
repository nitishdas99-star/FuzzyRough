use std::fs;
use std::path::{Path, PathBuf};

use approx::assert_abs_diff_eq;
use frlearn_core::Matrix;
use frlearn_neighbor::{BruteForceIndex, Metric, NeighborIndex};
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

fn matrix_f64(value: &Value) -> Matrix {
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

fn matrix_usize(value: &Value) -> Array2<usize> {
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
                .map(|v| v.as_u64().expect("matrix entry should be u64") as usize)
        })
        .collect::<Vec<_>>();
    Array2::from_shape_vec((n_rows, n_cols), flat).expect("matrix shape should be consistent")
}

#[test]
fn brute_force_query_matches_python_fixture() {
    let x_norm_fixture = read_json("x_norm.json");
    let knn_fixture = read_json("knn.json");

    let x_train_norm = matrix_f64(&x_norm_fixture["x_train_norm"]);
    let x_query_norm = matrix_f64(&x_norm_fixture["x_query_norm"]);
    let expected_indices = matrix_usize(&knn_fixture["indices"]);
    let expected_distances = matrix_f64(&knn_fixture["distances"]);
    let k = knn_fixture["k"].as_u64().expect("k should be u64") as usize;

    assert_eq!(
        knn_fixture["metric"]
            .as_str()
            .expect("metric should be string"),
        "euclidean"
    );

    let index = BruteForceIndex::new(x_train_norm, Metric::Euclidean);
    let (actual_indices, actual_distances) =
        index.query(&x_query_norm, k).expect("query should succeed");

    assert_eq!(actual_indices, expected_indices);
    assert_eq!(actual_distances.shape(), expected_distances.shape());
    for row_idx in 0..actual_distances.nrows() {
        for col_idx in 0..actual_distances.ncols() {
            assert_abs_diff_eq!(
                actual_distances[[row_idx, col_idx]],
                expected_distances[[row_idx, col_idx]],
                epsilon = 1e-10
            );
        }
    }
}
