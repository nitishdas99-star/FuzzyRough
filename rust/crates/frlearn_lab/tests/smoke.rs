use approx::assert_relative_eq;
use frlearn_lab::runner::{BackendChoice, DatasetChoice, LabConfig, SuiteChoice};

#[test]
fn smoke_classifier_suite_runs_and_metrics_sane() {
    let cfg = LabConfig {
        suite: SuiteChoice::Classifier,
        dataset: DatasetChoice::Overlap,
        backend: BackendChoice::Bruteforce,
        n_train: 200,
        n_test: 200,
        dims: 6,
        noise: 0.25,
        seed: 42,
        k: 5,
        use_frfs: true,
        use_frps: true,
    };

    let rep = frlearn_lab::runner::run(cfg);
    assert!(!rep.models.is_empty());

    for m in rep.models {
        assert!((0.0..=1.0).contains(&m.accuracy));
        assert!((0.0..=1.0).contains(&m.macro_f1));
        // confusion matrix rows sum to n_test
        let total: u64 = m.confusion.iter().sum();
        assert_eq!(total as usize, 200);
    }
}

#[test]
fn smoke_novelty_suite_runs_and_auc_sane() {
    let cfg = LabConfig {
        suite: SuiteChoice::Novelty,
        dataset: DatasetChoice::Overlap,
        backend: BackendChoice::Bruteforce,
        n_train: 200,
        n_test: 300,
        dims: 5,
        noise: 0.25,
        seed: 7,
        k: 5,
        use_frfs: false,
        use_frps: false,
    };

    let rep = frlearn_lab::runner::run(cfg);
    let novelty = rep.novelty.expect("expected novelty report");
    assert_eq!(novelty.n_train_inliers, 200);
    assert_eq!(novelty.n_test_total, 600);
    assert!(!novelty.descriptors.is_empty());

    for d in novelty.descriptors {
        assert!((0.0..=1.0).contains(&d.roc_auc));
        assert!((0.0..=1.0).contains(&d.tpr_at_fpr_1pct));
        assert!((0.0..=1.0).contains(&d.tpr_at_fpr_5pct));
    }
}

#[test]
fn metrics_roc_auc_perfect_separation_is_one() {
    let y = vec![0u8, 0, 1, 1];
    let scores = vec![0.1, 0.2, 0.8, 0.9];
    let auc = frlearn_lab::metrics::roc_auc(&y, &scores);
    assert_relative_eq!(auc, 1.0, epsilon = 1e-12);
}
