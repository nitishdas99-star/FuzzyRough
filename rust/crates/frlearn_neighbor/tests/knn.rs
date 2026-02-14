use approx::assert_abs_diff_eq;
use frlearn_neighbor::{BruteForceIndex, Metric, NeighborIndex};
use ndarray::array;

#[test]
fn euclidean_knn_returns_known_neighbors() {
    let x_train = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let x_query = array![[0.1, 0.1], [1.5, 1.5]];

    let index = BruteForceIndex::new(x_train, Metric::Euclidean);
    let (indices, distances) = index.query(&x_query, 2).expect("query should succeed");

    assert_eq!(indices.shape(), &[2, 2]);
    assert_eq!(distances.shape(), &[2, 2]);
    assert_eq!(indices[[0, 0]], 0);
    assert_eq!(indices[[0, 1]], 1);
    assert_eq!(indices[[1, 0]], 1);
    assert_eq!(indices[[1, 1]], 2);

    assert_abs_diff_eq!(distances[[0, 0]], (0.02f64).sqrt(), epsilon = 1e-12);
    assert_abs_diff_eq!(distances[[1, 0]], (0.5f64).sqrt(), epsilon = 1e-12);
}

#[test]
fn manhattan_metric_is_supported() {
    let x_train = array![[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]];
    let x_query = array![[1.0, 1.0]];

    let index = BruteForceIndex::new(x_train, Metric::Manhattan);
    let (indices, distances) = index.query(&x_query, 2).expect("query should succeed");

    assert_eq!(indices[[0, 0]], 1);
    assert_eq!(indices[[0, 1]], 0);
    assert_abs_diff_eq!(distances[[0, 0]], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(distances[[0, 1]], 2.0, epsilon = 1e-12);
}

#[test]
fn query_clamps_k_to_train_size_and_handles_empty_queries() {
    let x_train = array![[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]];
    let x_query = array![[0.2, 0.2]];
    let empty_query = ndarray::Array2::<f64>::zeros((0, 2));

    let index = BruteForceIndex::new(x_train, Metric::Euclidean);

    let (indices, distances) = index.query(&x_query, 10).expect("query should succeed");
    assert_eq!(indices.shape(), &[1, 3]);
    assert_eq!(distances.shape(), &[1, 3]);

    let (empty_indices, empty_distances) =
        index.query(&empty_query, 5).expect("query should succeed");
    assert_eq!(empty_indices.shape(), &[0, 3]);
    assert_eq!(empty_distances.shape(), &[0, 3]);
}

#[test]
fn distances_are_sorted_per_query_row() {
    let x_train = array![[2.0, 2.0], [0.0, 0.0], [1.0, 1.0], [3.0, 3.0]];
    let x_query = array![[0.9, 0.9], [2.1, 2.1]];

    let index = BruteForceIndex::new(x_train, Metric::Euclidean);
    let (_indices, distances) = index.query(&x_query, 4).expect("query should succeed");

    for row in distances.rows() {
        for col in 1..row.len() {
            assert!(row[col - 1] <= row[col]);
        }
    }
}
