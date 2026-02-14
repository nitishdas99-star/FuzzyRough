use approx::assert_abs_diff_eq;
use frlearn_math::{
    complement, decreasing_weights, l1_norm, l2_distance, l2_norm, lukasiewicz_t_norm,
    lukasiewicz_t_norm_scalar, max_norm, min_t_norm, min_t_norm_scalar, product_t_norm,
    product_t_norm_scalar, safe_divide, safe_normalize_rows, uniform_weights,
};
use ndarray::array;

#[test]
fn t_norm_scalars_stay_inside_unit_interval() {
    let samples = [0.0, 0.1, 0.3, 0.5, 0.9, 1.0];
    for a in samples {
        for b in samples {
            let min_value = min_t_norm_scalar(a, b);
            let product_value = product_t_norm_scalar(a, b);
            let luka_value = lukasiewicz_t_norm_scalar(a, b);

            assert!((0.0..=1.0).contains(&min_value));
            assert!((0.0..=1.0).contains(&product_value));
            assert!((0.0..=1.0).contains(&luka_value));
        }
    }
}

#[test]
fn t_norm_boundary_conditions_hold() {
    let samples = [0.0, 0.2, 0.5, 0.8, 1.0];
    for a in samples {
        assert_abs_diff_eq!(min_t_norm_scalar(a, 1.0), a, epsilon = 1e-12);
        assert_abs_diff_eq!(product_t_norm_scalar(a, 1.0), a, epsilon = 1e-12);
        assert_abs_diff_eq!(lukasiewicz_t_norm_scalar(a, 1.0), a, epsilon = 1e-12);

        assert_abs_diff_eq!(min_t_norm_scalar(a, 0.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(product_t_norm_scalar(a, 0.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(lukasiewicz_t_norm_scalar(a, 0.0), 0.0, epsilon = 1e-12);
    }
}

#[test]
fn elementwise_t_norms_work_and_validate_shapes() {
    let left = array![[0.1, 0.3], [0.7, 1.0]];
    let right = array![[0.9, 0.2], [0.4, 1.0]];
    let mismatch = array![[0.5, 0.5, 0.5]];

    let min_result = min_t_norm(&left, &right).expect("same shapes");
    let product_result = product_t_norm(&left, &right).expect("same shapes");
    let luka_result = lukasiewicz_t_norm(&left, &right).expect("same shapes");

    assert_abs_diff_eq!(min_result[[0, 0]], 0.1, epsilon = 1e-12);
    assert_abs_diff_eq!(product_result[[0, 1]], 0.06, epsilon = 1e-12);
    assert_abs_diff_eq!(luka_result[[1, 0]], 0.1, epsilon = 1e-12);

    assert!(min_t_norm(&left, &mismatch).is_err());
    assert!(product_t_norm(&left, &mismatch).is_err());
    assert!(lukasiewicz_t_norm(&left, &mismatch).is_err());
}

#[test]
fn normalize_rows_sums_to_one_without_nans() {
    let matrix = array![
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-3.0, 2.0, f64::NAN],
        [f64::INFINITY, -2.0, f64::NEG_INFINITY]
    ];

    let normalized = safe_normalize_rows(&matrix);
    for row in normalized.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-12);
        assert!(row.iter().all(|value| !value.is_nan()));
    }
}

#[test]
fn weight_vectors_are_normalized() {
    let uniform = uniform_weights(4);
    let decreasing = decreasing_weights(4);

    assert_abs_diff_eq!(uniform.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(decreasing.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
    assert!(decreasing.windows(2).all(|pair| pair[0] >= pair[1]));
}

#[test]
fn vector_measures_return_expected_values() {
    let row = array![3.0, -4.0, 0.0];
    assert_abs_diff_eq!(l1_norm(row.view()), 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(l2_norm(row.view()), 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(max_norm(row.view()), 4.0, epsilon = 1e-12);

    let left = array![0.0, 0.0, 0.0];
    let right = array![3.0, 4.0, 0.0];
    let distance = l2_distance(left.view(), right.view()).expect("same length");
    assert_abs_diff_eq!(distance, 5.0, epsilon = 1e-12);
}

#[test]
fn transformations_are_safe_for_non_finite_values() {
    assert_abs_diff_eq!(complement(0.25), 0.75, epsilon = 1e-12);
    assert_abs_diff_eq!(complement(2.0), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(safe_divide(3.0, 0.0, -1.0), -1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(safe_divide(f64::NAN, 2.0, 9.0), 9.0, epsilon = 1e-12);
}
