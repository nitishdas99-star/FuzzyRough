use frlearn_core::Matrix;
use frlearn_neighbor::{BruteForceIndex, Metric, NeighborIndex};
use std::hint::black_box;
use std::time::Instant;

fn synthetic_matrix(rows: usize, cols: usize, scale: f64) -> Matrix {
    let mut values = Vec::with_capacity(rows * cols);
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            values.push(((row_idx * 17 + col_idx * 13) as f64).sin() * scale);
        }
    }
    Matrix::from_shape_vec((rows, cols), values).expect("shape is valid")
}

fn run_benchmark() {
    let x_train = synthetic_matrix(512, 16, 1.0);
    let x_query = synthetic_matrix(64, 16, 0.5);
    let index = BruteForceIndex::new(x_train, Metric::Euclidean);
    let iterations = 100usize;

    let start = Instant::now();
    for _iteration in 0..iterations {
        let _ = index
            .query(black_box(&x_query), black_box(10))
            .expect("query should succeed");
    }
    println!(
        "bruteforce_query_64x512_k10: {} iterations in {:?}",
        iterations,
        start.elapsed()
    );
}

fn main() {
    run_benchmark();
}
