use frlearn_core::{Estimator, Matrix, Predictor};
use frlearn_neighbor::Metric;
use frlearn_neighbours::FRNN;
use std::hint::black_box;
use std::time::Instant;

fn synthetic_matrix(rows: usize, cols: usize, scale: f64) -> Matrix {
    let mut values = Vec::with_capacity(rows * cols);
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            values.push((((row_idx * 31 + col_idx * 7) % 97) as f64 / 97.0) * scale);
        }
    }
    Matrix::from_shape_vec((rows, cols), values).expect("shape is valid")
}

fn synthetic_labels(rows: usize, classes: usize) -> Vec<usize> {
    (0..rows).map(|row_idx| row_idx % classes).collect()
}

fn run_benchmark() {
    let x_train = synthetic_matrix(600, 12, 1.0);
    let y_train = synthetic_labels(600, 3);
    let x_query = synthetic_matrix(120, 12, 0.9);

    let model = FRNN {
        k: 7,
        metric: Metric::Euclidean,
    }
    .fit(&x_train, &y_train)
    .expect("fit should succeed");
    let iterations = 80usize;

    let start = Instant::now();
    for _iteration in 0..iterations {
        let _ = model
            .predict_scores(black_box(&x_query))
            .expect("predict_scores should succeed");
    }
    println!(
        "frnn_predict_scores_120x600_k7: {} iterations in {:?}",
        iterations,
        start.elapsed()
    );
}

fn main() {
    run_benchmark();
}
