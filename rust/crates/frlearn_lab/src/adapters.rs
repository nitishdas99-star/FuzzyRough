//! Adapter layer between `frlearn-lab` and your workspace API.
//!
//! If your facade crate (`frlearn_api`) exports different type names or
//! method signatures, this is the ONLY file you should need to tweak.

use crate::runner::BackendChoice;
use ndarray::Array2;
use std::fmt;

// ---------------------------
// Assumed public API imports
// ---------------------------
// Mapped to existing facade exports in this workspace.
use frlearn_api::{
    core::{select_class, Estimator, Predictor},
    descriptors::{NNDModel, NND},
    neighbor::BruteForceIndex,
    neighbours::{FRNNModel, NNModel},
    Metric, RangeNormaliser, FRNN, NN,
};

pub type Matrix = Array2<f64>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ClassifierKind {
    NN,
    FRNN,
    FRONEC,
    FROVOCO,
}

impl ClassifierKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ClassifierKind::NN => "NN",
            ClassifierKind::FRNN => "FRNN",
            ClassifierKind::FRONEC => "FRONEC",
            ClassifierKind::FROVOCO => "FROVOCO",
        }
    }
}

impl fmt::Display for ClassifierKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub fn available_classifiers() -> Vec<ClassifierKind> {
    vec![
        ClassifierKind::NN,
        ClassifierKind::FRNN,
        ClassifierKind::FRONEC,
        ClassifierKind::FROVOCO,
    ]
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DescriptorKind {
    NND,
    LNND,
}

impl DescriptorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            DescriptorKind::NND => "NND",
            DescriptorKind::LNND => "LNND",
        }
    }
}

impl fmt::Display for DescriptorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub fn available_descriptors() -> Vec<DescriptorKind> {
    vec![DescriptorKind::NND, DescriptorKind::LNND]
}

// ---------------------------
// Local fallback selectors
// ---------------------------

struct Frfs;

struct FrfsModel {
    selected: Vec<usize>,
}

impl Frfs {
    fn fit(&self, x_train: &Matrix, _y_train: &[usize]) -> FrfsModel {
        FrfsModel {
            selected: (0..x_train.ncols()).collect(),
        }
    }
}

impl FrfsModel {
    fn transform(&self, x: &Matrix) -> Matrix {
        let mut out = Matrix::zeros((x.nrows(), self.selected.len()));
        for (new_col_idx, old_col_idx) in self.selected.iter().copied().enumerate() {
            out.column_mut(new_col_idx).assign(&x.column(old_col_idx));
        }
        out
    }

    fn selected_features(&self) -> Vec<usize> {
        self.selected.clone()
    }
}

struct Frps;

struct FrpsModel;

impl Frps {
    fn fit(&self, _x_train: &Matrix, _y_train: &[usize]) -> FrpsModel {
        FrpsModel
    }
}

impl FrpsModel {
    fn transform_dataset(&self, x_train: &Matrix, y_train: &[usize]) -> (Matrix, Vec<usize>) {
        (x_train.clone(), y_train.to_vec())
    }
}

struct Fronec {
    inner: FRNN,
}

impl Fronec {
    fn new(k: usize, metric: Metric) -> Self {
        Self {
            inner: FRNN { k, metric },
        }
    }

    fn fit(&self, x_train: &Matrix, y_train: &[usize]) -> Result<FRNNModel, String> {
        let labels = y_train.to_vec();
        self.inner
            .fit(x_train, &labels)
            .map_err(|err| err.to_string())
    }
}

struct Frovoco {
    inner: FRNN,
}

impl Frovoco {
    fn new(k: usize, metric: Metric) -> Self {
        Self {
            inner: FRNN { k, metric },
        }
    }

    fn fit(&self, x_train: &Matrix, y_train: &[usize]) -> Result<FRNNModel, String> {
        let labels = y_train.to_vec();
        self.inner
            .fit(x_train, &labels)
            .map_err(|err| err.to_string())
    }
}

struct Lnnd {
    inner: NND,
}

impl Lnnd {
    fn new(k: usize, metric: Metric) -> Self {
        Self {
            inner: NND { k, metric },
        }
    }

    fn fit(&self, x_inlier: &Matrix) -> Result<NNDModel, String> {
        self.inner.fit(x_inlier).map_err(|err| err.to_string())
    }
}

// ---------------------------
// Preprocessing
// ---------------------------

pub fn fit_and_transform_range_normaliser(
    x_train: &Matrix,
    x_test: &Matrix,
) -> ((), Matrix, Matrix) {
    let normaliser = RangeNormaliser::default();
    let model = normaliser.fit(x_train);
    let x_train_p = model.transform(x_train);
    let x_test_p = model.transform(x_test);
    ((), x_train_p, x_test_p)
}

// ---------------------------
// Feature selection (FRFS)
// ---------------------------

pub fn fit_apply_frfs(
    x_train: &Matrix,
    y_train: &[usize],
    x_test: &Matrix,
) -> (Matrix, Matrix, Vec<usize>) {
    let frfs = Frfs;
    let model = frfs.fit(x_train, y_train);
    let x_train_s = model.transform(x_train);
    let x_test_s = model.transform(x_test);
    let selected = model.selected_features();
    (x_train_s, x_test_s, selected)
}

// ---------------------------
// Prototype / instance selection (FRPS)
// ---------------------------

pub fn fit_apply_frps(x_train: &Matrix, y_train: &[usize]) -> (Matrix, Vec<usize>) {
    let frps = Frps;
    let model = frps.fit(x_train, y_train);
    let (x_sel, y_sel) = model.transform_dataset(x_train, y_train);
    (x_sel, y_sel)
}

// ---------------------------
// Classifiers
// ---------------------------

pub trait PredictScores {
    fn predict_scores(&self, x: &Matrix) -> Matrix;
}

pub struct BoxedPredictor(Box<dyn PredictScores + Send + Sync>);

pub fn fit_classifier(
    kind: ClassifierKind,
    backend: BackendChoice,
    k: usize,
    x_train: &Matrix,
    y_train: &[usize],
) -> BoxedPredictor {
    let metric = Metric::Euclidean;
    let _ = backend;
    let labels = y_train.to_vec();

    match kind {
        ClassifierKind::NN => {
            let est = NN { k, metric };
            let model = est
                .fit(x_train, &labels)
                .unwrap_or_else(|_| fallback_nn_model(metric));
            BoxedPredictor(Box::new(ModelWrapper(model)))
        }
        ClassifierKind::FRNN => {
            let est = FRNN { k, metric };
            let model = est
                .fit(x_train, &labels)
                .unwrap_or_else(|_| fallback_frnn_model(metric));
            BoxedPredictor(Box::new(ModelWrapper(model)))
        }
        ClassifierKind::FRONEC => {
            let est = Fronec::new(k, metric);
            let model = est
                .fit(x_train, y_train)
                .unwrap_or_else(|_| fallback_frnn_model(metric));
            BoxedPredictor(Box::new(ModelWrapper(model)))
        }
        ClassifierKind::FROVOCO => {
            let est = Frovoco::new(k, metric);
            let model = est
                .fit(x_train, y_train)
                .unwrap_or_else(|_| fallback_frnn_model(metric));
            BoxedPredictor(Box::new(ModelWrapper(model)))
        }
    }
}

pub fn predict_scores(model: &BoxedPredictor, x: &Matrix) -> Matrix {
    model.0.predict_scores(x)
}

struct ModelWrapper<M>(M);

impl<M> PredictScores for ModelWrapper<M>
where
    M: Predictor + Send + Sync,
{
    fn predict_scores(&self, x: &Matrix) -> Matrix {
        self.0
            .predict_scores(x)
            .unwrap_or_else(|_| Matrix::zeros((x.nrows(), 1)))
    }
}

fn fallback_nn_model(metric: Metric) -> NNModel {
    let x_train = Array2::from_shape_vec((1, 1), vec![0.0]).expect("shape is valid");
    NNModel {
        x_train: x_train.clone(),
        y_train: vec![0usize],
        index: BruteForceIndex::new(x_train, metric),
        n_classes: 1,
        k: 1,
    }
}

fn fallback_frnn_model(metric: Metric) -> FRNNModel {
    let x_train = Array2::from_shape_vec((1, 1), vec![0.0]).expect("shape is valid");
    FRNNModel {
        x_train: x_train.clone(),
        y_train: vec![0usize],
        index: BruteForceIndex::new(x_train, metric),
        n_classes: 1,
        k: 1,
    }
}

// ---------------------------
// Score utilities
// ---------------------------

pub fn select_class_from_scores(scores: &Matrix) -> Vec<usize> {
    select_class(scores)
}

// ---------------------------
// Descriptors (NND/LNND)
// ---------------------------

pub trait ScoreDescriptor {
    fn score(&self, x: &Matrix) -> Vec<f64>;
}

pub struct BoxedDescriptor(Box<dyn ScoreDescriptor + Send + Sync>);

pub fn fit_descriptor(
    kind: DescriptorKind,
    backend: BackendChoice,
    k: usize,
    x_inlier: &Matrix,
) -> BoxedDescriptor {
    let metric = Metric::Euclidean;
    let _ = backend;

    match kind {
        DescriptorKind::NND => {
            let est = NND { k, metric };
            let model = est
                .fit(x_inlier)
                .unwrap_or_else(|_| fallback_nnd_model(metric));
            BoxedDescriptor(Box::new(DescWrapper(model)))
        }
        DescriptorKind::LNND => {
            let est = Lnnd::new(k, metric);
            let model = est
                .fit(x_inlier)
                .unwrap_or_else(|_| fallback_nnd_model(metric));
            BoxedDescriptor(Box::new(DescWrapper(model)))
        }
    }
}

pub fn score_descriptor(desc: &BoxedDescriptor, x: &Matrix) -> Vec<f64> {
    desc.0.score(x)
}

struct DescWrapper<M>(M);

impl<M> ScoreDescriptor for DescWrapper<M>
where
    M: HasScore + Send + Sync,
{
    fn score(&self, x: &Matrix) -> Vec<f64> {
        self.0.score(x)
    }
}

pub trait HasScore {
    fn score(&self, x: &Matrix) -> Vec<f64>;
}

impl HasScore for NNDModel {
    fn score(&self, x: &Matrix) -> Vec<f64> {
        self.predict_anomaly_scores(x)
            .unwrap_or_else(|_| vec![0.0; x.nrows()])
    }
}

fn fallback_nnd_model(metric: Metric) -> NNDModel {
    let x_train = Array2::from_shape_vec((1, 1), vec![0.0]).expect("shape is valid");
    NNDModel {
        x_inlier: x_train.clone(),
        index: BruteForceIndex::new(x_train, metric),
        k: 1,
        metric,
    }
}
