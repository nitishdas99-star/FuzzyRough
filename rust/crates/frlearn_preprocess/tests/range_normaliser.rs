use approx::assert_abs_diff_eq;
use frlearn_core::Matrix;
use frlearn_preprocess::{RangeNormaliser, RangeNormaliserModel, TransformerModel};
use ndarray::array;

#[test]
fn range_normaliser_matches_known_small_matrix_output() {
    let x_train = array![[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]];
    let normaliser = RangeNormaliser { eps: 0.0 };
    let model = normaliser.fit(&x_train);

    let transformed = model.transform(&x_train);
    let expected = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]];

    for row_idx in 0..expected.nrows() {
        for col_idx in 0..expected.ncols() {
            assert_abs_diff_eq!(
                transformed[[row_idx, col_idx]],
                expected[[row_idx, col_idx]],
                epsilon = 1e-12
            );
        }
    }
}

#[test]
fn constant_column_does_not_produce_nans() {
    let x_train = array![[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]];
    let model = RangeNormaliser::default().fit(&x_train);
    let transformed = model.transform(&x_train);

    assert!(transformed.iter().all(|value| !value.is_nan()));
    assert!(transformed.column(0).iter().all(|value| *value == 0.0));
}

#[test]
fn in_range_data_stays_inside_unit_interval() {
    let x_train = array![[0.0, 10.0], [10.0, 20.0], [5.0, 15.0]];
    let x_query = array![[0.0, 10.0], [4.0, 13.0], [10.0, 20.0]];
    let model = RangeNormaliser::default().fit(&x_train);
    let transformed = model.transform(&x_query);

    for value in transformed.iter() {
        assert!((0.0..=1.0).contains(value));
    }
}

#[test]
fn transformer_trait_returns_error_on_feature_mismatch() {
    let x_train = array![[1.0, 2.0], [3.0, 4.0]];
    let x_bad = Matrix::zeros((2, 3));

    let model = RangeNormaliser::default().fit(&x_train);
    let result = <RangeNormaliserModel as TransformerModel>::transform(&model, &x_bad);

    assert!(result.is_err());
}
