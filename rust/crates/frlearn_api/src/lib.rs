//! Facade crate exposing the main FuzzyRough APIs.
//!
//! # Examples
//!
//! Classification pipeline on synthetic data:
//! ```rust
//! use frlearn_api::{default_frnn_pipeline, Metric};
//!
//! let x_train = frlearn_api::core::Matrix::from_shape_vec(
//!     (6, 2),
//!     vec![0.0, 0.0, 0.2, 0.1, 4.8, 5.0, 5.1, 4.9, 9.9, 10.2, 10.1, 9.8],
//! )?;
//! let y_train = vec![0usize, 0, 1, 1, 2, 2];
//! let x_query = frlearn_api::core::Matrix::from_shape_vec(
//!     (3, 2),
//!     vec![0.1, 0.0, 5.0, 5.0, 10.0, 10.0],
//! )?;
//!
//! let pipeline = default_frnn_pipeline(3, Metric::Euclidean);
//! let model = pipeline.fit(&x_train, &y_train)?;
//! let labels = model.predict(&x_query)?;
//! let probabilities = model.predict_proba(&x_query)?;
//!
//! assert_eq!(labels.len(), x_query.nrows());
//! assert_eq!(probabilities.nrows(), x_query.nrows());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Novelty scoring example:
//! ```rust
//! use frlearn_api::{descriptors::NND, Metric};
//!
//! let inliers = frlearn_api::core::Matrix::from_shape_vec(
//!     (4, 2),
//!     vec![0.0, 0.1, 0.2, -0.1, -0.1, 0.0, 0.1, 0.2],
//! )?;
//! let mixed = frlearn_api::core::Matrix::from_shape_vec(
//!     (4, 2),
//!     vec![0.0, 0.0, 0.3, -0.2, 4.0, 4.2, 5.5, 5.1],
//! )?;
//!
//! let descriptor = NND { k: 2, metric: Metric::Euclidean };
//! let model = descriptor.fit(&inliers)?;
//! let scores = model.predict_anomaly_scores(&mixed)?;
//! assert_eq!(scores.len(), mixed.nrows());
//! assert!(scores.iter().all(|value| value.is_finite()));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
#![cfg_attr(doc, warn(missing_docs))]
#![warn(rust_2018_idioms)]
#![warn(clippy::all)]

pub use frlearn_core as core;
pub use frlearn_descriptors as descriptors;
pub use frlearn_math as math;
pub use frlearn_neighbor as neighbor;
pub use frlearn_neighbor::{BruteForceIndex, Metric};
pub use frlearn_neighbours as neighbours;
pub use frlearn_neighbours::{FRNN, NN};
pub use frlearn_preprocess as preprocess;
pub use frlearn_preprocess::RangeNormaliser;

use frlearn_core::{
    Estimator, FrResult, Labels, Matrix, Predictor, probabilities_from_scores, select_class,
};
use frlearn_preprocess::{RangeNormaliserModel, TransformerModel};

pub type PredictProbaOutput = Matrix;

pub struct Pipeline<C> {
    pub normaliser: RangeNormaliser,
    pub classifier: C,
}

impl<C> Pipeline<C>
where
    C: Estimator,
{
    pub fn fit(&self, x: &Matrix, y: &Labels) -> FrResult<PipelineModel<C::Model>> {
        let normaliser_model = self.normaliser.fit(x);
        let normalized_train =
            <RangeNormaliserModel as TransformerModel>::transform(&normaliser_model, x)?;
        let classifier_model = self.classifier.fit(&normalized_train, y)?;

        Ok(PipelineModel {
            normaliser_model,
            classifier_model,
        })
    }
}

pub struct PipelineModel<M> {
    pub normaliser_model: RangeNormaliserModel,
    pub classifier_model: M,
}

impl<M> PipelineModel<M>
where
    M: Predictor,
{
    pub fn predict_scores(&self, xq: &Matrix) -> FrResult<Matrix> {
        let normalized_query =
            <RangeNormaliserModel as TransformerModel>::transform(&self.normaliser_model, xq)?;
        self.classifier_model.predict_scores(&normalized_query)
    }

    pub fn predict(&self, xq: &Matrix) -> FrResult<Labels> {
        let scores = self.predict_scores(xq)?;
        Ok(select_class(&scores))
    }

    pub fn predict_proba(&self, xq: &Matrix) -> FrResult<PredictProbaOutput> {
        let scores = self.predict_scores(xq)?;
        Ok(probabilities_from_scores(&scores))
    }
}

pub fn default_nn_pipeline(k: usize, metric: Metric) -> Pipeline<NN> {
    Pipeline {
        normaliser: RangeNormaliser::default(),
        classifier: NN { k, metric },
    }
}

pub fn default_frnn_pipeline(k: usize, metric: Metric) -> Pipeline<FRNN> {
    Pipeline {
        normaliser: RangeNormaliser::default(),
        classifier: FRNN { k, metric },
    }
}

pub fn workspace_status() -> &'static str {
    "frlearn rust workspace initialized"
}
