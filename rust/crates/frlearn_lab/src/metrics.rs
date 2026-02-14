use ndarray::Array2;

pub fn accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let mut correct = 0usize;
    for (a, b) in y_true.iter().zip(y_pred.iter()) {
        if a == b {
            correct += 1;
        }
    }
    correct as f64 / y_true.len() as f64
}

pub fn confusion_matrix(y_true: &[usize], y_pred: &[usize]) -> Array2<u64> {
    let c = 1 + y_true
        .iter()
        .chain(y_pred.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let mut m = Array2::<u64>::zeros((c, c));
    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        m[(*t, *p)] += 1;
    }
    m
}

pub fn macro_f1_and_confusion(y_true: &[usize], y_pred: &[usize]) -> (f64, Array2<u64>) {
    let conf = confusion_matrix(y_true, y_pred);
    let c = conf.nrows();
    let mut f1_sum = 0.0;
    for cls in 0..c {
        let tp = conf[(cls, cls)] as f64;
        let fp: f64 = (0..c).map(|r| conf[(r, cls)] as f64).sum::<f64>() - tp;
        let fn_: f64 = (0..c).map(|c2| conf[(cls, c2)] as f64).sum::<f64>() - tp;
        let precision = if (tp + fp) > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if (tp + fn_) > 0.0 {
            tp / (tp + fn_)
        } else {
            0.0
        };
        let f1 = if (precision + recall) > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        f1_sum += f1;
    }
    let macro_f1 = if c > 0 { f1_sum / c as f64 } else { 0.0 };
    (macro_f1, conf)
}

/// ROC-AUC for binary labels (0=inlier/negative, 1=outlier/positive).
/// Uses trapezoidal integration over the ROC curve.
pub fn roc_auc(y_true: &[u8], scores: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let mut pairs: Vec<(f64, u8)> = scores.iter().copied().zip(y_true.iter().copied()).collect();
    // sort by decreasing score
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let positives = y_true.iter().filter(|&&y| y == 1).count() as f64;
    let negatives = y_true.iter().filter(|&&y| y == 0).count() as f64;
    if positives == 0.0 || negatives == 0.0 {
        return 0.0;
    }

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_tpr = 0.0;
    let mut prev_fpr = 0.0;
    let mut auc = 0.0;

    for (_s, y) in pairs {
        if y == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let tpr = tp / positives;
        let fpr = fp / negatives;
        // trapezoid between (prev_fpr,prev_tpr) and (fpr,tpr)
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    auc.clamp(0.0, 1.0)
}

/// Compute TPR at a target FPR by sweeping thresholds.
pub fn tpr_at_fpr(y_true: &[u8], scores: &[f64], target_fpr: f64) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let mut pairs: Vec<(f64, u8)> = scores.iter().copied().zip(y_true.iter().copied()).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let positives = y_true.iter().filter(|&&y| y == 1).count() as f64;
    let negatives = y_true.iter().filter(|&&y| y == 0).count() as f64;
    if positives == 0.0 || negatives == 0.0 {
        return 0.0;
    }

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut best_tpr = 0.0;

    for (_s, y) in pairs {
        if y == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let tpr = tp / positives;
        let fpr = fp / negatives;
        if fpr <= target_fpr {
            best_tpr = tpr;
        } else {
            break;
        }
    }
    best_tpr.clamp(0.0, 1.0)
}
