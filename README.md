# FuzzyRough

Rust fuzzy-rough learning workspace for classification and novelty detection. This repository is an in-progress Rust implementation/port of fuzzy-rough learning concepts, with composable crates for preprocessing, math primitives, neighbour search, neighbour-based classifiers, descriptor models, and a synthetic-data lab app for reproducible benchmarking.

## 1) Project Title + Short Description

`FuzzyRough` is a multi-crate Rust workspace focused on fuzzy-rough learning workflows. It currently includes reusable protocol traits, matrix/math helpers, `RangeNormaliser`, neighbour search, neighbour classifiers (`NN`, `FRNN`), descriptor models (`NND` with `LNND` adapter path in lab), score utilities, and the `frlearn_lab` CLI that runs synthetic benchmark suites across preprocessing, feature/prototype selection adapters, classification, and novelty detection.

## 2) Features

- Core protocols: `Estimator`, `Predictor`, typed matrix/label aliases, and `FrError`.
- Score utilities: `select_class` and `probabilities_from_scores` (handles zero rows and non-finite inputs safely).
- Preprocessing: `RangeNormaliser` with fit/model/transform flow.
- Fuzzy math primitives: t-norms, transformations, safe row normalization, weights, norm/distance helpers.
- Neighbour search: `Metric` (`Euclidean`, `Manhattan`) and `BruteForceIndex` with sorted top-k output.
- Classifiers: `NN` and `FRNN` implemented in `frlearn_neighbours`.
- Descriptors: `NND` implemented; `LNND` available in `frlearn_lab` adapter layer.
- Lab compatibility adapters: `FRFS`, `FRPS`, `FRONEC`, `FROVOCO`, `LNND` paths are currently mapped as placeholders/wrappers in `frlearn_lab/src/adapters.rs`.
- Feature-gated descriptors/backends: `svm`, `eif`, `sae`, `kdtree`, `balltree` are exposed as Cargo features; several are currently placeholder passthroughs.
- Python-reference fixture workflow: `tools/export_fixtures.py` generates JSON fixtures consumed by Rust tests.

## 3) Workspace Layout

Workspace members from `rust/Cargo.toml`:

- `crates/frlearn_core`: core traits and error/types.
  - Entry points: `Estimator`, `Predictor`, `FrError`, `Matrix`, `Labels`, `select_class`, `probabilities_from_scores`.
- `crates/frlearn_math`: fuzzy/math helpers used by other crates.
  - Entry points: `min_t_norm_scalar`, `product_t_norm_scalar`, `lukasiewicz_t_norm_scalar`, `safe_normalize_rows`, `uniform_weights`, `decreasing_weights`, `l1_norm`, `l2_norm`.
- `crates/frlearn_preprocess`: preprocessing models.
  - Entry points: `RangeNormaliser`, `RangeNormaliserModel`, `TransformerModel`.
- `crates/frlearn_neighbor`: neighbour index abstraction and brute-force backend.
  - Entry points: `Metric`, `pairwise_distance`, `NeighborIndex`, `BruteForceIndex`.
- `crates/frlearn_neighbours`: neighbour-based classifiers.
  - Entry points: `NN`, `NNModel`, `FRNN`, `FRNNModel`.
- `crates/frlearn_descriptors`: descriptor models.
  - Entry points: `NND`, `NNDModel`; feature-gated placeholder modules `svm`, `eif`, `sae`.
- `crates/frlearn_api`: public facade crate and pipeline API.
  - Entry points: re-exports of core/math/preprocess/neighbor/neighbours/descriptors, plus `Pipeline`, `PipelineModel`, `default_nn_pipeline`, `default_frnn_pipeline`.
- `crates/frlearn_lab`: CLI benchmark harness over synthetic datasets.
  - Entry points: `main.rs` CLI, `runner::run`, `report` serialization/printing, `datasets`, `metrics`, `adapters`.

## 4) Dependencies

Dependency data below is extracted from each crate `Cargo.toml`.

### `frlearn_core`

| dependency | version/source | optional? | purpose |
| --- | --- | --- | --- |
| `ndarray` | `0.16.1` (workspace) | No | matrix type alias (`Array2<f64>`) |
| `thiserror` | `2.0.17` (workspace) | No | typed error definitions |
| `approx` (dev) | `0.5.1` (workspace) | No | floating-point assertions in tests |

### `frlearn_math`

| dependency | version/source | optional? | purpose |
| --- | --- | --- | --- |
| `ndarray` | `0.16.1` (workspace) | No | array operations for t-norms/transforms |
| `thiserror` | `2.0.17` (workspace) | No | `MathError` |
| `approx` (dev) | `0.5.1` (workspace) | No | numeric test tolerances |

### `frlearn_preprocess`

| dependency | version/source | optional? | purpose |
| --- | --- | --- | --- |
| `ndarray` | `0.16.1` (workspace) | No | min/max vectors and matrix transforms |
| `frlearn_core` | `path = "../frlearn_core"` | No | shared `Matrix`, `FrError`, `FrResult` |
| `frlearn_math` | `path = "../frlearn_math"` | No | `clamp01`, `safe_divide` |
| `approx` (dev) | `0.5.1` (workspace) | No | floating-point test checks |
| `serde_json` (dev) | `1.0` (workspace) | No | loading Python fixture JSON in tests |

### `frlearn_neighbor`

| dependency | version/source | optional? | purpose |
| --- | --- | --- | --- |
| `ndarray` | `0.16.1` (workspace) | No | index/distance output arrays |
| `frlearn_core` | `path = "../frlearn_core"` | No | shared error and matrix types |
| `frlearn_math` | `path = "../frlearn_math"` | No | shared numeric helpers |
| `approx` (dev) | `0.5.1` (workspace) | No | distance comparisons in tests |
| `serde_json` (dev) | `1.0` (workspace) | No | Python fixture loading |

### `frlearn_neighbours`

| dependency | version/source | optional? | purpose |
| --- | --- | --- | --- |
| `ndarray` | `0.16.1` (workspace) | No | score matrices |
| `frlearn_core` | `path = "../frlearn_core"` | No | estimator/predictor traits and errors |
| `frlearn_neighbor` | `path = "../frlearn_neighbor"` | No | KNN query backend |
| `frlearn_preprocess` | `path = "../frlearn_preprocess"` | No | preprocessing interoperability |
| `frlearn_math` | `path = "../frlearn_math"` | No | fuzzy transformations and normalization |
| `approx` (dev) | `0.5.1` (workspace) | No | score tolerance checks |
| `serde_json` (dev) | `1.0` (workspace) | No | Python fixture loading |

### `frlearn_descriptors`

| dependency | version/source | optional? | purpose |
| --- | --- | --- | --- |
| `ndarray` | `0.16.1` (workspace) | No | score matrix output |
| `frlearn_core` | `path = "../frlearn_core"` | No | common types and predictor trait |
| `frlearn_neighbor` | `path = "../frlearn_neighbor"` | No | neighbour queries for NND |
| `frlearn_math` | `path = "../frlearn_math"` | No | safe averaging/normalization helpers |
| `frlearn_preprocess` | `path = "../frlearn_preprocess"` | No | preprocessing interoperability |
| `approx` (dev) | `0.5.1` (workspace) | No | descriptor test tolerances |

### `frlearn_api`

| dependency | version/source | optional? | purpose |
| --- | --- | --- | --- |
| `serde` | `1.0` (workspace, `derive`) | No | facade-side serialization support |
| `serde_json` | `1.0` (workspace) | No | JSON helpers in facade consumers |
| `frlearn_core` | `path = "../frlearn_core"` | No | core protocol re-export |
| `frlearn_math` | `path = "../frlearn_math"` | No | math re-export |
| `frlearn_preprocess` | `path = "../frlearn_preprocess"` | No | preprocess re-export |
| `frlearn_neighbor` | `path = "../frlearn_neighbor"` | No | neighbour re-export |
| `frlearn_neighbours` | `path = "../frlearn_neighbours"` | No | classifier re-export |
| `frlearn_descriptors` | `path = "../frlearn_descriptors"` | No | descriptor re-export |
| `approx` (dev) | `0.5.1` (workspace) | No | e2e probability checks |
| `ndarray` (dev) | `0.16.1` (workspace) | No | test arrays/macros |

### `frlearn_lab`

| dependency | version/source | optional? | purpose |
| --- | --- | --- | --- |
| `clap` | `4.5` + `derive` | No | CLI parsing |
| `rand` | `0.8` | No | synthetic data RNG |
| `rand_distr` | `0.4` | No | Gaussian/uniform sampling |
| `serde` | `1` + `derive` | No | report serialization |
| `serde_json` | `1` | No | JSON report output |
| `ndarray` | `0.16` + `serde` | No | dataset and confusion matrix storage |
| `approx` | `0.5` | No | test assertions |
| `frlearn_api` | `path = "../frlearn_api"` | No | workspace facade integration |

Platform notes:

- The workspace currently uses pure Rust dependencies; no BLAS/LAPACK setup is required.
- Windows builds are supported and commonly used in this repository.

## 5) Installation / Build

Windows PowerShell:

```bash
cd D:\FuzzyRough\rust
cargo build
cargo test --workspace
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
```

Generic shell:

```bash
cd rust
cargo build
cargo test --workspace
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
```

Required environment variables: none.

Supply-chain checks:

```bash
cd D:\FuzzyRough\rust
cargo audit
cargo deny check
```

## 6) Quick Start (Library Usage)

### A) Classification pipeline (`RangeNormaliser` + optional FRFS/FRPS + `FRNN`)

```rust
use frlearn_api::{
    core::{probabilities_from_scores, select_class, Estimator, Predictor},
    Metric, RangeNormaliser, FRNN,
};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x_train = array![
        [0.0, 0.0],
        [0.2, 0.1],
        [4.8, 5.0],
        [5.1, 4.9],
        [9.9, 10.2],
        [10.1, 9.8],
    ];
    let y_train = vec![0usize, 0, 1, 1, 2, 2];
    let x_query = array![[0.1, 0.0], [5.0, 5.1], [10.0, 10.0]];

    let normaliser = RangeNormaliser::default();
    let norm_model = normaliser.fit(&x_train);
    let x_train_n = norm_model.transform(&x_train);
    let x_query_n = norm_model.transform(&x_query);

    // Optional FRFS/FRPS path in this workspace currently lives in frlearn_lab adapters.
    // Those adapters are currently compatibility placeholders (identity selection).
    let (x_train_fs, x_query_fs, _selected) =
        frlearn_lab::adapters::fit_apply_frfs(&x_train_n, &y_train, &x_query_n);
    let (x_train_ps, y_train_ps) = frlearn_lab::adapters::fit_apply_frps(&x_train_fs, &y_train);

    let estimator = FRNN {
        k: 3,
        metric: Metric::Euclidean,
    };
    let model = estimator.fit(&x_train_ps, &y_train_ps)?;
    let scores = model.predict_scores(&x_query_fs)?;
    let y_pred = select_class(&scores);
    let proba = probabilities_from_scores(&scores);

    println!("predictions = {y_pred:?}");
    println!("probabilities = {proba:?}");
    Ok(())
}
```

### B) Novelty detection (`NND`)

```rust
use frlearn_api::{descriptors::NND, Metric};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x_inlier = array![
        [0.0, 0.1],
        [0.2, -0.1],
        [-0.1, 0.0],
        [0.1, 0.2],
    ];
    let x_mixed = array![
        [0.0, 0.0],   // likely inlier
        [0.3, -0.2],  // likely inlier
        [4.0, 4.2],   // likely outlier
        [5.5, 5.1],   // likely outlier
    ];

    let descriptor = NND {
        k: 2,
        metric: Metric::Euclidean,
    };
    let model = descriptor.fit(&x_inlier)?;
    let scores = model.predict_anomaly_scores(&x_mixed)?;

    let threshold = 0.6_f64;
    let is_outlier: Vec<bool> = scores.iter().map(|s| *s >= threshold).collect();
    println!("scores = {scores:?}");
    println!("is_outlier = {is_outlier:?}");
    Ok(())
}
```

## 7) CLI Quick Start (`frlearn_lab`)

Run all suites:

```bash
cd D:\FuzzyRough\rust
cargo run -p frlearn_lab -- --suite all
```

Example with explicit tuning flags and JSON output:

```bash
cargo run -p frlearn_lab -- --suite classifier --dataset overlap --seed 7 --dims 12 --n-train 1500 --n-test 1000 --noise 0.2 --k 7 --out results.json
```

Suites:

- `all`: runs classifier + novelty pipelines and prints both reports.
- `classifier`: evaluates `NN`, `FRNN`, and adapter-routed `FRONEC`/`FROVOCO` on selected synthetic dataset.
- `novelty`: evaluates descriptors (`NND`, adapter-routed `LNND`) with ROC-AUC/TPR@FPR metrics.

Backend consistency notes:

- Current default backend is `bruteforce`.
- `kdtree`/`balltree` enums are behind features and currently compile-time hooks.

Flag note:

- `--label-noise` is not present in the current CLI implementation. Available noise control is `--noise`.

## 8) Applications

### Predictive maintenance

- Industrial telemetry is often noisy and partially ambiguous; fuzzy-rough scoring handles uncertain boundaries naturally.
- `RangeNormaliser` + neighbour classifiers can provide interpretable class confidence over machine states.
- Novelty workflows (`NND`) are useful for early anomaly screening when fault labels are sparse.

### Medical diagnosis

- Clinical data is heterogeneous and uncertain; fuzzy similarity supports soft decisions instead of brittle thresholds.
- Class probability outputs from `predict_scores` + `probabilities_from_scores` support triage/risk ranking.
- Feature-focused workflows can be layered via FRFS/FRPS adapters while native implementations continue to expand.

### Remote sensing / land-cover mapping

- Spectral signatures overlap heavily across classes; FRNN-style lower/upper approximations help with mixed pixels.
- Brute-force KNN and deterministic synthetic tests support method validation before scaling.
- Normalization + neighbour scoring is robust for prototyping small-to-medium geospatial subsets.

### Cybersecurity anomaly and intrusion detection

- One-class/novelty settings are common when attack labels are limited.
- `NND` provides distance-based anomaly scores and thresholding primitives.
- Deterministic seed-based lab datasets enable repeatable benchmarking of detector behavior.

### Genomics / biomarker discovery

- High-dimensional noisy feature spaces benefit from conservative preprocessing and similarity-based classifiers.
- FRNN’s class-wise fuzzy approximations can improve ranking stability under uncertain labels.
- Prototype and feature-selection adapter paths help design future sparse/high-dimensional pipelines.

### Quality control in manufacturing

- Small shifts in sensor distributions can be captured with neighbour-based local models.
- Descriptor scores can flag abnormal production batches early.
- The lab CLI enables stress testing on overlap/redundant/prototype-heavy synthetic regimes before deployment.

## 9) File-by-File / Module Documentation

```text
FuzzyRough/
├── LICENSE
├── README.md
├── rust/
│   ├── Cargo.toml
│   ├── Cargo.lock
│   ├── PORTING_PLAN.md
│   ├── fixtures/python_reference/
│   │   ├── x_norm.json
│   │   ├── knn.json
│   │   └── scores.json
│   └── crates/
│       ├── frlearn_core/
│       │   ├── src/{lib.rs,error.rs,types.rs}
│       │   └── tests/core_contract.rs
│       ├── frlearn_math/
│       │   ├── src/{lib.rs,t_norms.rs,transformations.rs,weights.rs,vector_measures.rs}
│       │   └── tests/math_props.rs
│       ├── frlearn_preprocess/
│       │   ├── src/{lib.rs,range_normaliser.rs}
│       │   └── tests/{range_normaliser.rs,python_fixtures.rs}
│       ├── frlearn_neighbor/
│       │   ├── src/{lib.rs,distance.rs,bruteforce.rs}
│       │   └── tests/{knn.rs,python_fixtures.rs}
│       ├── frlearn_neighbours/
│       │   ├── src/{lib.rs,nn.rs,frnn.rs}
│       │   └── tests/{nn.rs,frnn_shapes.rs,python_fixtures.rs}
│       ├── frlearn_descriptors/
│       │   ├── src/{lib.rs,nnd.rs}
│       │   └── tests/nnd.rs
│       ├── frlearn_api/
│       │   ├── src/lib.rs
│       │   ├── examples/iris_nn.rs
│       │   └── tests/e2e.rs
│       └── frlearn_lab/
│           ├── src/{lib.rs,main.rs,runner.rs,datasets.rs,adapters.rs,metrics.rs,report.rs}
│           └── tests/smoke.rs
└── tools/
    └── export_fixtures.py
```

Module notes:

- `rust/Cargo.toml`: workspace membership, edition, and shared dependency versions.
- `rust/PORTING_PLAN.md`: staged module roadmap (P0..P6) for Python-to-Rust parity work.
- `tools/export_fixtures.py`: generates deterministic JSON fixtures (`x_norm`, `knn`, `scores`) used in Rust integration tests.
- `frlearn_lab/src/runner.rs`: suite orchestration and timing collection.
- `frlearn_lab/src/datasets.rs`: synthetic datasets A–E for class overlap, XOR, redundant features, prototype pressure, and one-class novelty.
- `frlearn_lab/src/adapters.rs`: compatibility layer mapping lab workloads to current facade exports; includes placeholder selectors/wrappers.
- `frlearn_lab/src/metrics.rs`: classification and novelty metrics (`accuracy`, `macro_f1`, `roc_auc`, `tpr_at_fpr`).
- `frlearn_lab/src/report.rs`: report structs + console/JSON rendering.

## 10) Testing

- Unit tests validate per-module behavior (e.g., t-norm boundaries, normalization safety, KNN sorted distances).
- Integration tests validate end-to-end crate contracts (`frlearn_api/tests/e2e.rs`, `frlearn_lab/tests/smoke.rs`).
- Python-reference fixture tests compare Rust outputs to deterministic JSON fixtures from `tools/export_fixtures.py`.
- Synthetic data generation uses explicit seeds (`--seed` in lab CLI) for reproducible runs.
- Full validation command:

```bash
cd D:\FuzzyRough\rust
cargo test --workspace
```

## 11) Feature Flags

| crate | feature | enables | extra dependencies |
| --- | --- | --- | --- |
| `frlearn_descriptors` | `svm` | placeholder `svm` descriptor module | none currently |
| `frlearn_descriptors` | `eif` | placeholder `eif` descriptor module | none currently |
| `frlearn_descriptors` | `sae` | placeholder `sae` descriptor module | none currently |
| `frlearn_api` | `svm` | forwards to `frlearn_descriptors/svm` | none currently |
| `frlearn_api` | `eif` | forwards to `frlearn_descriptors/eif` | none currently |
| `frlearn_api` | `sae` | forwards to `frlearn_descriptors/sae` | none currently |
| `frlearn_api` | `kdtree` | backend feature hook | none currently |
| `frlearn_api` | `balltree` | backend feature hook | none currently |
| `frlearn_lab` | `svm` | forwards to `frlearn_api/svm` | none currently |
| `frlearn_lab` | `eif` | forwards to `frlearn_api/eif` | none currently |
| `frlearn_lab` | `sae` | forwards to `frlearn_api/sae` | none currently |
| `frlearn_lab` | `kdtree` | forwards to `frlearn_api/kdtree` | none currently |
| `frlearn_lab` | `balltree` | forwards to `frlearn_api/balltree` | none currently |

## 12) Versioning + MSRV

- Workspace version: `0.1.0`.
- Rust editions:
  - Workspace crates: `edition = "2024"` (from workspace package config).
  - `frlearn_lab`: `edition = "2021"` (explicit crate setting).
- MSRV: not pinned yet (`rust-version` is not set in the manifests).

## 13) License

License file is present at `LICENSE` in the repository root.

## 14) Contributing

Standard contribution flow:

1. Fork and clone your branch of the repository.
2. Create a topic branch (`feat/...`, `fix/...`, `docs/...`).
3. Make focused changes with tests.
4. Run:
   - `cargo fmt`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo test --workspace`
5. Open a pull request with clear scope and test evidence.

### Porting policy

When porting modules from Python to Rust, keep dependency order and parity discipline:

1. Build shared core contracts first (`frlearn_core`).
2. Add math primitives (`frlearn_math`) before algorithm crates.
3. Add preprocessors before classifiers/descriptors.
4. Add neighbour backends before neighbour-based models.
5. Add algorithms incrementally with passing unit + integration tests.
6. Add/refresh golden fixtures (`tools/export_fixtures.py`) and compare with tolerant numeric assertions.
7. Keep Python reference behavior intact while extending Rust coverage.

## 15) Citation

If you use this toolkit in research, cite:

```bibtex
@software{das2026fuzzyrough,
  author = {Nitish Das},
  title = {FuzzyRough (Rust fuzzy-rough learning toolkit)},
  year = {2026},
  note = {Rust workspace for fuzzy-rough classification and novelty detection}
}
```

## 16) Contact / Author

Maintained by Nitish Das.
Email ID: nitishdas99@gmail.com
