use crate::runner::{BackendChoice, DatasetChoice, SuiteChoice};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimingReport {
    pub preprocess_ms: u64,
    pub frfs_ms: u64,
    pub frps_ms: u64,
    pub train_ms: u64,
    pub predict_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelReport {
    pub model: String,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub confusion: Array2<u64>,
    pub timing: TimingReport,
    pub n_train_used: usize,
    pub dims_used: usize,
    pub selected_features: Option<Vec<usize>>,
}

impl ModelReport {
    pub fn new(model: String) -> Self {
        Self {
            model,
            accuracy: 0.0,
            macro_f1: 0.0,
            confusion: Array2::<u64>::zeros((0, 0)),
            timing: TimingReport::default(),
            n_train_used: 0,
            dims_used: 0,
            selected_features: None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NoveltyReport {
    pub preprocess_ms: u64,
    pub n_train_inliers: usize,
    pub n_test_total: usize,
    pub descriptors: Vec<DescriptorReport>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DescriptorReport {
    pub name: String,
    pub roc_auc: f64,
    pub tpr_at_fpr_1pct: f64,
    pub tpr_at_fpr_5pct: f64,
    pub fit_ms: u64,
    pub score_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuiteReport {
    pub suite: SuiteChoice,
    pub dataset: DatasetChoice,
    pub backend: BackendChoice,
    pub models: Vec<ModelReport>,
    pub novelty: Option<NoveltyReport>,
}

impl SuiteReport {
    pub fn new(suite: SuiteChoice, dataset: DatasetChoice, backend: BackendChoice) -> Self {
        Self {
            suite,
            dataset,
            backend,
            models: Vec::new(),
            novelty: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonReport {
    pub suite: String,
    pub dataset: String,
    pub backend: String,
    pub models: Vec<ModelReport>,
    pub novelty: Option<NoveltyReport>,
}

impl JsonReport {
    pub fn from_suite(r: &SuiteReport) -> Self {
        Self {
            suite: format!("{:?}", r.suite),
            dataset: format!("{:?}", r.dataset),
            backend: format!("{:?}", r.backend),
            models: r.models.clone(),
            novelty: r.novelty.clone(),
        }
    }
}

pub fn print_suite_report(r: &SuiteReport) {
    println!("\n=== frlearn-lab ===");
    println!("Suite:   {:?}", r.suite);
    println!("Dataset: {:?}", r.dataset);
    println!("Backend: {:?}", r.backend);

    if !r.models.is_empty() {
        println!("\n--- Classification ---");
        println!(
            "{:<10} {:>8} {:>8} | {:>6} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6}",
            "model", "acc", "mF1", "prep", "frfs", "frps", "train", "pred", "n_tr", "dims"
        );
        for m in &r.models {
            println!(
                "{:<10} {:>8.4} {:>8.4} | {:>6} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6}",
                m.model,
                m.accuracy,
                m.macro_f1,
                m.timing.preprocess_ms,
                m.timing.frfs_ms,
                m.timing.frps_ms,
                m.timing.train_ms,
                m.timing.predict_ms,
                m.n_train_used,
                m.dims_used,
            );
        }
        println!(
            "\nTip: add --out results.json to save full reports (including confusion matrices).\n"
        );
    }

    if let Some(n) = &r.novelty {
        println!("\n--- Novelty / One-class ---");
        println!("Timing(ms): preprocess={}", n.preprocess_ms);
        println!(
            "n_train_inliers={} n_test_total={}",
            n.n_train_inliers, n.n_test_total
        );
        println!(
            "\n{:<8} {:>8} {:>10} {:>10} | {:>6} {:>6}",
            "desc", "auc", "tpr@1%", "tpr@5%", "fit", "score"
        );
        for d in &n.descriptors {
            println!(
                "{:<8} {:>8.4} {:>10.4} {:>10.4} | {:>6} {:>6}",
                d.name, d.roc_auc, d.tpr_at_fpr_1pct, d.tpr_at_fpr_5pct, d.fit_ms, d.score_ms
            );
        }
        println!();
    }
}
