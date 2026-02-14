pub fn uniform_weights(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    vec![1.0 / n as f64; n]
}

pub fn decreasing_weights(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    let denominator = (n * (n + 1) / 2) as f64;
    (1..=n)
        .rev()
        .map(|weight| weight as f64 / denominator)
        .collect()
}
