use crate::adapters;
use crate::datasets;
use crate::metrics;
use crate::report::{DescriptorReport, ModelReport, NoveltyReport, SuiteReport, TimingReport};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum SuiteChoice {
    All,
    Classifier,
    Novelty,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum DatasetChoice {
    Overlap,
    Xor,
    Redundant,
    PrototypeHeavy,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum BackendChoice {
    Bruteforce,
    #[cfg(feature = "kdtree")]
    Kdtree,
    #[cfg(feature = "balltree")]
    Balltree,
}

#[derive(Clone, Debug)]
pub struct LabConfig {
    pub suite: SuiteChoice,
    pub dataset: DatasetChoice,
    pub backend: BackendChoice,
    pub n_train: usize,
    pub n_test: usize,
    pub dims: usize,
    pub noise: f64,
    pub seed: u64,
    pub k: usize,
    pub use_frfs: bool,
    pub use_frps: bool,
}

pub fn run(cfg: LabConfig) -> SuiteReport {
    let mut report = SuiteReport::new(cfg.suite, cfg.dataset, cfg.backend);

    match cfg.suite {
        SuiteChoice::All => {
            report.models.extend(run_classifier_suite(&cfg));
            report.novelty = Some(run_novelty_suite(&cfg));
        }
        SuiteChoice::Classifier => {
            report.models.extend(run_classifier_suite(&cfg));
        }
        SuiteChoice::Novelty => {
            report.novelty = Some(run_novelty_suite(&cfg));
        }
    }

    report
}

fn run_classifier_suite(cfg: &LabConfig) -> Vec<ModelReport> {
    let ds = match cfg.dataset {
        DatasetChoice::Overlap => {
            datasets::overlapping_gaussians(cfg.n_train, cfg.n_test, cfg.dims, cfg.noise, cfg.seed)
        }
        DatasetChoice::Xor => datasets::xor_with_distractors(
            cfg.n_train,
            cfg.n_test,
            cfg.dims.max(2),
            cfg.noise,
            cfg.seed,
        ),
        DatasetChoice::Redundant => datasets::redundant_irrelevant(
            cfg.n_train,
            cfg.n_test,
            cfg.dims.max(20),
            cfg.noise,
            cfg.seed,
        ),
        DatasetChoice::PrototypeHeavy => datasets::prototype_heavy(
            cfg.n_train.max(3000),
            cfg.n_test,
            cfg.dims,
            cfg.noise,
            cfg.seed,
        ),
    };

    let mut x_train = ds.x_train;
    let mut y_train = ds.y_train;
    let x_test = ds.x_test;
    let y_test = ds.y_test;

    let mut timing = TimingReport::default();

    // Preprocess
    let t0 = Instant::now();
    let (_prep_model, x_train_p, x_test_p) =
        adapters::fit_and_transform_range_normaliser(&x_train, &x_test);
    timing.preprocess_ms = t0.elapsed().as_millis() as u64;
    x_train = x_train_p;

    // Feature selection
    let mut selected_features: Option<Vec<usize>> = None;
    let mut x_test_fs = x_test_p;
    if cfg.use_frfs {
        let t1 = Instant::now();
        let (x_train_fs, x_test_fs2, selected) =
            adapters::fit_apply_frfs(&x_train, &y_train, &x_test_fs);
        timing.frfs_ms = t1.elapsed().as_millis() as u64;
        x_train = x_train_fs;
        x_test_fs = x_test_fs2;
        selected_features = Some(selected);
    }

    // Prototype selection
    if cfg.use_frps {
        let t2 = Instant::now();
        let (x_train_ps, y_train_ps) = adapters::fit_apply_frps(&x_train, &y_train);
        timing.frps_ms = t2.elapsed().as_millis() as u64;
        x_train = x_train_ps;
        y_train = y_train_ps;
    }

    // Train + evaluate each classifier
    let mut reports = Vec::new();

    for model_kind in adapters::available_classifiers() {
        let t_train = Instant::now();
        let model = adapters::fit_classifier(model_kind, cfg.backend, cfg.k, &x_train, &y_train);
        let train_ms = t_train.elapsed().as_millis() as u64;

        let t_pred = Instant::now();
        let scores = adapters::predict_scores(&model, &x_test_fs);
        let pred_ms = t_pred.elapsed().as_millis() as u64;

        let y_pred = adapters::select_class_from_scores(&scores);
        let acc = metrics::accuracy(&y_test, &y_pred);
        let (macro_f1, conf) = metrics::macro_f1_and_confusion(&y_test, &y_pred);

        let mut m = ModelReport::new(model_kind.to_string());
        m.accuracy = acc;
        m.macro_f1 = macro_f1;
        m.confusion = conf;
        m.timing = timing.clone();
        m.timing.train_ms = train_ms;
        m.timing.predict_ms = pred_ms;
        m.n_train_used = x_train.nrows();
        m.dims_used = x_train.ncols();
        m.selected_features = selected_features.clone();

        reports.push(m);
    }

    reports
}

fn run_novelty_suite(cfg: &LabConfig) -> NoveltyReport {
    let ds = datasets::one_class_novelty(
        cfg.n_train,
        cfg.n_test * 2,
        cfg.dims.max(2),
        cfg.noise,
        cfg.seed,
    );

    let t0 = Instant::now();
    let (_prep_model, x_train_p, x_test_p) =
        adapters::fit_and_transform_range_normaliser(&ds.x_train, &ds.x_test);
    let preprocess_ms = t0.elapsed().as_millis() as u64;

    let mut rep = NoveltyReport {
        preprocess_ms,
        n_train_inliers: x_train_p.nrows(),
        n_test_total: x_test_p.nrows(),
        ..NoveltyReport::default()
    };

    for dk in adapters::available_descriptors() {
        let t1 = Instant::now();
        let descriptor = adapters::fit_descriptor(dk, cfg.backend, cfg.k, &x_train_p);
        let fit_ms = t1.elapsed().as_millis() as u64;

        let t2 = Instant::now();
        let scores = adapters::score_descriptor(&descriptor, &x_test_p);
        let score_ms = t2.elapsed().as_millis() as u64;

        let auc = metrics::roc_auc(&ds.y_test_binary, &scores);
        let tpr_at_1 = metrics::tpr_at_fpr(&ds.y_test_binary, &scores, 0.01);
        let tpr_at_5 = metrics::tpr_at_fpr(&ds.y_test_binary, &scores, 0.05);

        let dr = DescriptorReport {
            name: dk.to_string(),
            roc_auc: auc,
            tpr_at_fpr_1pct: tpr_at_1,
            tpr_at_fpr_5pct: tpr_at_5,
            fit_ms,
            score_ms,
        };
        rep.descriptors.push(dr);
    }

    rep
}
