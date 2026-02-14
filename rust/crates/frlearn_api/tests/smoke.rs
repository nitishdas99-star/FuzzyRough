use approx::assert_abs_diff_eq;
use frlearn_api::{Metric, default_nn_pipeline};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn make_dataset(
    seed: u64,
) -> (
    frlearn_api::core::Matrix,
    Vec<usize>,
    frlearn_api::core::Matrix,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let class_centers = [(0.0_f64, 0.0_f64), (5.0_f64, 5.0_f64), (10.0_f64, 10.0_f64)];
    let samples_per_class = 8usize;

    let mut train = Vec::with_capacity(class_centers.len() * samples_per_class * 2);
    let mut labels = Vec::with_capacity(class_centers.len() * samples_per_class);
    for (class_idx, (cx, cy)) in class_centers.iter().copied().enumerate() {
        for _sample_idx in 0..samples_per_class {
            train.push(cx + rng.gen_range(-0.25..0.25));
            train.push(cy + rng.gen_range(-0.25..0.25));
            labels.push(class_idx);
        }
    }

    let query = vec![
        0.1, -0.1, //
        5.1, 4.9, //
        10.2, 9.9, //
        2.5, 2.5, //
    ];

    let x_train =
        frlearn_api::core::Matrix::from_shape_vec((labels.len(), 2), train).expect("shape valid");
    let x_query = frlearn_api::core::Matrix::from_shape_vec((4, 2), query).expect("shape valid");
    (x_train, labels, x_query)
}

#[test]
fn smoke_pipeline_outputs_well_formed_probabilities() {
    for seed in [7_u64, 42_u64, 99_u64] {
        let (x_train, y_train, x_query) = make_dataset(seed);
        let pipeline = default_nn_pipeline(3, Metric::Euclidean);
        let model = pipeline
            .fit(&x_train, &y_train)
            .expect("fit should succeed");

        let labels = model.predict(&x_query).expect("predict should succeed");
        let probabilities = model
            .predict_proba(&x_query)
            .expect("predict_proba should succeed");

        assert_eq!(labels.len(), x_query.nrows());
        assert_eq!(probabilities.nrows(), x_query.nrows());
        for row in probabilities.rows() {
            assert!(row.iter().all(|value| value.is_finite()));
            assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-12);
        }
    }
}
