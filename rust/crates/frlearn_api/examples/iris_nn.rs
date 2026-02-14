use frlearn_api::{Metric, default_nn_pipeline};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x_train = array![
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9]
    ];
    let y_train = vec![0usize, 0, 1, 1, 2, 2];

    let x_query = array![
        [5.0, 3.4, 1.5, 0.2],
        [6.5, 3.0, 5.2, 2.0],
        [6.1, 2.9, 4.7, 1.4]
    ];

    let pipeline = default_nn_pipeline(3, Metric::Euclidean);
    let model = pipeline.fit(&x_train, &y_train)?;

    let predictions = model.predict(&x_query)?;
    let probabilities = model.predict_proba(&x_query)?;

    println!("Predicted classes: {:?}", predictions);
    println!("Class probabilities:\n{probabilities:?}");

    Ok(())
}
