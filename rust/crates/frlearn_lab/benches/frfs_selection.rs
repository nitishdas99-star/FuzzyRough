use frlearn_lab::adapters::fit_apply_frfs;
use ndarray::Array2;
use std::hint::black_box;
use std::time::Instant;

fn synthetic_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut values = Vec::with_capacity(rows * cols);
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            values.push(((row_idx * 11 + col_idx * 19) as f64).sin());
        }
    }
    Array2::from_shape_vec((rows, cols), values).expect("shape is valid")
}

fn synthetic_labels(rows: usize, classes: usize) -> Vec<usize> {
    (0..rows).map(|row_idx| row_idx % classes).collect()
}

fn run_benchmark() {
    let x_train = synthetic_matrix(800, 24);
    let y_train = synthetic_labels(800, 4);
    let x_test = synthetic_matrix(200, 24);
    let iterations = 80usize;

    let start = Instant::now();
    for _iteration in 0..iterations {
        let _ = fit_apply_frfs(black_box(&x_train), black_box(&y_train), black_box(&x_test));
    }
    println!(
        "frfs_fit_apply_identity_800x24: {} iterations in {:?}",
        iterations,
        start.elapsed()
    );
}

fn main() {
    run_benchmark();
}
