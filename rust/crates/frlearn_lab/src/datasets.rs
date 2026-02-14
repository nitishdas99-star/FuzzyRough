use ndarray::Array2;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};

#[derive(Debug)]
pub struct ClassifDataset {
    pub x_train: Array2<f64>,
    pub y_train: Vec<usize>,
    pub x_test: Array2<f64>,
    pub y_test: Vec<usize>,
}

#[derive(Debug)]
pub struct NoveltyDataset {
    pub x_train: Array2<f64>,   // inliers only
    pub x_test: Array2<f64>,    // mix inliers+outliers
    pub y_test_binary: Vec<u8>, // 0=inlier,1=outlier
}

fn split_counts(total: usize, groups: usize) -> Vec<usize> {
    if groups == 0 {
        return Vec::new();
    }
    let base = total / groups;
    let remainder = total % groups;
    (0..groups)
        .map(|idx| base + usize::from(idx < remainder))
        .collect()
}

/// Dataset A: 3 strongly-overlapping Gaussians.
pub fn overlapping_gaussians(
    n_train: usize,
    n_test: usize,
    dims: usize,
    noise: f64,
    seed: u64,
) -> ClassifDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let classes = 3usize;
    let train_counts = split_counts(n_train, classes);
    let test_counts = split_counts(n_test, classes);

    let mut xtr = Array2::<f64>::zeros((n_train, dims));
    let mut ytr = vec![0usize; n_train];
    let mut xte = Array2::<f64>::zeros((n_test, dims));
    let mut yte = vec![0usize; n_test];

    // Center spacing controls overlap: smaller => more overlap.
    let center_spacing = 0.75;
    let base = Normal::new(0.0, 1.0).unwrap();
    let noise_d = Normal::new(0.0, noise.max(0.0)).unwrap();

    let mut train_offset = 0usize;
    let mut test_offset = 0usize;
    for c in 0..classes {
        let shift = (c as f64 - 1.0) * center_spacing;
        for i in 0..train_counts[c] {
            let row = train_offset + i;
            for j in 0..dims {
                let v = base.sample(&mut rng) + shift + noise_d.sample(&mut rng);
                xtr[(row, j)] = v;
            }
            ytr[row] = c;
        }
        for i in 0..test_counts[c] {
            let row = test_offset + i;
            for j in 0..dims {
                let v = base.sample(&mut rng) + shift + noise_d.sample(&mut rng);
                xte[(row, j)] = v;
            }
            yte[row] = c;
        }
        train_offset += train_counts[c];
        test_offset += test_counts[c];
    }

    // Shuffle rows to remove ordering artifacts.
    shuffle_in_unison(&mut rng, &mut xtr, &mut ytr);
    shuffle_in_unison(&mut rng, &mut xte, &mut yte);

    ClassifDataset {
        x_train: xtr,
        y_train: ytr,
        x_test: xte,
        y_test: yte,
    }
}

/// Dataset B: XOR in first 2 dims + distractor noise dims.
pub fn xor_with_distractors(
    n_train: usize,
    n_test: usize,
    dims: usize,
    noise: f64,
    seed: u64,
) -> ClassifDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let uni = Uniform::new(-1.0, 1.0);
    let noise_d = Normal::new(0.0, noise.max(0.0)).unwrap();

    let make = |n: usize, rng: &mut StdRng| {
        let mut x = Array2::<f64>::zeros((n, dims));
        let mut y = vec![0usize; n];
        for i in 0..n {
            let a = uni.sample(rng);
            let b = uni.sample(rng);
            let xa = a + noise_d.sample(rng);
            let xb = b + noise_d.sample(rng);
            x[(i, 0)] = xa;
            x[(i, 1)] = xb;
            // XOR label: different signs => class 1
            y[i] = ((xa >= 0.0) ^ (xb >= 0.0)) as usize;
            for j in 2..dims {
                x[(i, j)] = uni.sample(rng) + noise_d.sample(rng);
            }
        }
        (x, y)
    };

    let (mut xtr, mut ytr) = make(n_train, &mut rng);
    let (mut xte, mut yte) = make(n_test, &mut rng);
    shuffle_in_unison(&mut rng, &mut xtr, &mut ytr);
    shuffle_in_unison(&mut rng, &mut xte, &mut yte);

    ClassifDataset {
        x_train: xtr,
        y_train: ytr,
        x_test: xte,
        y_test: yte,
    }
}

/// Dataset C: small number of informative features, redundant features, and many irrelevant features.
pub fn redundant_irrelevant(
    n_train: usize,
    n_test: usize,
    dims: usize,
    noise: f64,
    seed: u64,
) -> ClassifDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let informative = 5usize;
    let redundant = 10usize;
    let irrelevant = dims.saturating_sub(informative + redundant).max(10);

    let base = Normal::new(0.0, 1.0).unwrap();
    let noise_d = Normal::new(0.0, noise.max(0.0)).unwrap();

    fn make_set(
        rng: &mut StdRng,
        n: usize,
        informative: usize,
        redundant: usize,
        irrelevant: usize,
        base: Normal<f64>,
        noise_d: Normal<f64>,
    ) -> (Array2<f64>, Vec<usize>) {
        let total = informative + redundant + irrelevant;
        let mut x = Array2::<f64>::zeros((n, total));
        let mut y = vec![0usize; n];
        for i in 0..n {
            // latent signal
            let z = base.sample(rng);
            let label = (z > 0.0) as usize;
            y[i] = label;

            // informative: correlated with z
            for j in 0..informative {
                x[(i, j)] = z + 0.5 * base.sample(rng) + noise_d.sample(rng);
            }

            // redundant: linear combos of informative with small noise
            for j in 0..redundant {
                let src = j % informative;
                x[(i, informative + j)] =
                    0.8 * x[(i, src)] + 0.2 * base.sample(rng) + 0.1 * noise_d.sample(rng);
            }

            // irrelevant: pure noise
            for j in 0..irrelevant {
                x[(i, informative + redundant + j)] = base.sample(rng) + noise_d.sample(rng);
            }
        }
        (x, y)
    }

    let (mut xtr, mut ytr) = make_set(
        &mut rng,
        n_train,
        informative,
        redundant,
        irrelevant,
        base,
        noise_d,
    );
    let (mut xte, mut yte) = make_set(
        &mut rng,
        n_test,
        informative,
        redundant,
        irrelevant,
        base,
        noise_d,
    );
    shuffle_in_unison(&mut rng, &mut xtr, &mut ytr);
    shuffle_in_unison(&mut rng, &mut xte, &mut yte);

    ClassifDataset {
        x_train: xtr,
        y_train: ytr,
        x_test: xte,
        y_test: yte,
    }
}

/// Dataset D: large clustered dataset to stress prototype selection.
pub fn prototype_heavy(
    n_train: usize,
    n_test: usize,
    dims: usize,
    noise: f64,
    seed: u64,
) -> ClassifDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let classes = 4usize;
    let clusters_per_class = 3usize;

    let train_counts = split_counts(n_train, classes);
    let test_counts = split_counts(n_test, classes);

    let mut xtr = Array2::<f64>::zeros((n_train, dims));
    let mut ytr = vec![0usize; n_train];
    let mut xte = Array2::<f64>::zeros((n_test, dims));
    let mut yte = vec![0usize; n_test];

    let base = Normal::new(0.0, 1.0).unwrap();
    let noise_d = Normal::new(0.0, noise.max(0.0)).unwrap();

    let mut train_offset = 0usize;
    let mut test_offset = 0usize;
    for c in 0..classes {
        // multiple cluster centers per class
        let mut centers = Vec::new();
        for k in 0..clusters_per_class {
            let mut center = vec![0.0; dims];
            for value in center.iter_mut().take(dims) {
                *value = (c as f64) * 1.5 + (k as f64) * 0.7 + 0.2 * base.sample(&mut rng);
            }
            centers.push(center);
        }

        for i in 0..train_counts[c] {
            let row = train_offset + i;
            let which = rng.gen_range(0..clusters_per_class);
            for j in 0..dims {
                xtr[(row, j)] =
                    centers[which][j] + base.sample(&mut rng) + noise_d.sample(&mut rng);
            }
            ytr[row] = c;
        }
        for i in 0..test_counts[c] {
            let row = test_offset + i;
            let which = rng.gen_range(0..clusters_per_class);
            for j in 0..dims {
                xte[(row, j)] =
                    centers[which][j] + base.sample(&mut rng) + noise_d.sample(&mut rng);
            }
            yte[row] = c;
        }
        train_offset += train_counts[c];
        test_offset += test_counts[c];
    }

    shuffle_in_unison(&mut rng, &mut xtr, &mut ytr);
    shuffle_in_unison(&mut rng, &mut xte, &mut yte);

    ClassifDataset {
        x_train: xtr,
        y_train: ytr,
        x_test: xte,
        y_test: yte,
    }
}

/// Dataset E: one-class novelty — train on inliers, test on mix of inliers+outliers.
pub fn one_class_novelty(
    n_train_inlier: usize,
    n_test_total: usize,
    dims: usize,
    noise: f64,
    seed: u64,
) -> NoveltyDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let inlier = Normal::new(0.0, 1.0).unwrap();
    let jitter = Normal::new(0.0, noise.max(0.0)).unwrap();

    // Train inliers
    let mut x_train = Array2::<f64>::zeros((n_train_inlier, dims));
    for i in 0..n_train_inlier {
        for j in 0..dims {
            x_train[(i, j)] = inlier.sample(&mut rng) + jitter.sample(&mut rng);
        }
    }

    // Test set: half inliers, half outliers
    let n_in = n_test_total / 2;
    let n_out = n_test_total - n_in;

    let mut x_test = Array2::<f64>::zeros((n_test_total, dims));
    let mut y = vec![0u8; n_test_total];

    // Inliers
    for i in 0..n_in {
        for j in 0..dims {
            x_test[(i, j)] = inlier.sample(&mut rng) + jitter.sample(&mut rng);
        }
        y[i] = 0;
    }

    // Outliers: uniform box far away + mild noise
    let uni = Uniform::new(3.0, 6.0);
    for i in 0..n_out {
        let row = n_in + i;
        for j in 0..dims {
            x_test[(row, j)] = uni.sample(&mut rng) + 0.25 * inlier.sample(&mut rng);
        }
        y[row] = 1;
    }

    // Shuffle test
    shuffle_novelty(&mut rng, &mut x_test, &mut y);

    NoveltyDataset {
        x_train,
        x_test,
        y_test_binary: y,
    }
}

fn shuffle_in_unison(rng: &mut StdRng, x: &mut Array2<f64>, y: &mut [usize]) {
    let n = x.nrows();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(rng);

    let x_old = x.clone();
    let y_old = y.to_owned();
    for (new_i, &old_i) in idx.iter().enumerate() {
        x.row_mut(new_i).assign(&x_old.row(old_i));
        y[new_i] = y_old[old_i];
    }
}

fn shuffle_novelty(rng: &mut StdRng, x: &mut Array2<f64>, y: &mut [u8]) {
    let n = x.nrows();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(rng);

    let x_old = x.clone();
    let y_old = y.to_owned();
    for (new_i, &old_i) in idx.iter().enumerate() {
        x.row_mut(new_i).assign(&x_old.row(old_i));
        y[new_i] = y_old[old_i];
    }
}
