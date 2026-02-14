# Rust Porting Plan (P0..P6)

## Scope note
The cloned repository at `D:\FuzzyRough` currently contains only `README.md` and `LICENSE`.
Python file mapping below is based on the provided snapshot file:
`C:\Users\Admin\Downloads\oulenz-fuzzy-rough-learn-8a5edab282632443 (1).txt`.

## Dependency order
1. P0 `frlearn_core`
2. P1 `frlearn_math`
3. P2 `frlearn_preprocess`
4. P3 `frlearn_neighbor`
5. P4 `frlearn_neighbours`
6. P5 `frlearn_descriptors`
7. P6 `frlearn_api`

## Module map

### P0 - `frlearn_core`
- Rust crate: `rust/frlearn_core`
- Python reference files:
  - `frlearn/base.py`
  - `frlearn/__init__.py`
  - `frlearn/tests/test_common.py`
- Expected I/O:
  - `Estimator::fit(X, y) -> Model`
  - `Model::predict_scores(Xq) -> Array2<f64>`
  - `Model::predict(Xq) -> Array1<usize>`
  - `select_class(scores) -> labels`
  - `probabilities_from_scores(scores) -> probabilities`
- Milestone tests:
  - unit tests for class selection and score-to-probability conversion
  - integration test for estimator/model contract

### P1 - `frlearn_math`
- Rust crate: `rust/frlearn_math`
- Python reference files:
  - `frlearn/array_functions.py`
  - `frlearn/dispersion_measures.py`
  - `frlearn/location_measures.py`
  - `frlearn/parametrisations.py`
  - `frlearn/t_norms.py`
  - `frlearn/transformations.py`
  - `frlearn/vector_size_measures.py`
  - `frlearn/weights.py`
- Expected I/O:
  - matrix/vector utilities, distances, norms, t-norms, transforms
- Milestone tests:
  - deterministic unit tests for each primitive
  - edge-case shape/NaN handling tests

### P2 - `frlearn_preprocess`
- Rust crate: `rust/frlearn_preprocess`
- Python reference files:
  - `frlearn/feature_preprocessors.py`
  - `frlearn/instance_preprocessors.py`
  - `frlearn/neighbours/feature_preprocessors.py`
  - `frlearn/neighbours/instance_preprocessors.py`
  - `frlearn/networks/feature_preprocessors.py`
- Expected I/O:
  - fit/transform preprocessors
  - feature and prototype selection pipelines
- Milestone tests:
  - unit tests for normalisation/selection behaviors
  - integration tests with small synthetic datasets

### P3 - `frlearn_neighbor`
- Rust crate: `rust/frlearn_neighbor`
- Python reference files:
  - `frlearn/neighbour_search_methods.py`
  - `frlearn/neighbours/neighbour_search_methods.py`
  - `frlearn/neighbours/utilities.py`
- Expected I/O:
  - nearest-neighbor index fit/query API
  - brute-force backend first, optional accelerated backends later
- Milestone tests:
  - exact neighbor order tests on toy data
  - cross-check against Python golden outputs for fixed seeds

### P4 - `frlearn_neighbours`
- Rust crate: `rust/frlearn_neighbours`
- Python reference files:
  - `frlearn/neighbours/classifiers.py`
  - `frlearn/neighbours/regressors.py`
  - `frlearn/neighbours/data_descriptors.py`
- Expected I/O:
  - neighbour-based classifiers/regressors and helper models
- Milestone tests:
  - parity tests for FRNN/NN-style methods on known fixtures
  - integration tests through core traits

### P5 - `frlearn_descriptors`
- Rust crate: `rust/frlearn_descriptors`
- Python reference files:
  - `frlearn/data_descriptors.py`
  - `frlearn/statistics/data_descriptors.py`
  - `frlearn/trees/data_descriptors.py`
  - `frlearn/support_vectors/data_descriptors.py`
- Expected I/O:
  - one-class scoring / novelty detection descriptors
- Milestone tests:
  - unit tests for each descriptor fit/infer path
  - parity tests where Python output fixtures are available

### P6 - `frlearn_api`
- Rust crate: `rust/frlearn_api`
- Python reference files:
  - `frlearn/classifiers.py`
  - `frlearn/regressors.py`
  - `frlearn/__init__.py`
- Expected I/O:
  - public facade and re-exports
  - examples and stable top-level API
- Milestone tests:
  - compile-only docs/examples
  - end-to-end integration tests from facade API

## Global test milestones per phase
After each phase P0..P6:
1. `cargo build --workspace`
2. `cargo test --workspace --all-targets`
3. `cargo test --workspace --tests`
4. Python parity comparison (when golden fixtures and Python sources are available)
5. `cargo fmt --all`
6. `cargo clippy --workspace --all-targets --all-features -- -D warnings`

## Optional feature flags
- `frlearn_descriptors/sae` (Shrink Autoencoder backend)
- `frlearn_descriptors/eif` (Extended Isolation Forest backend)
- `frlearn_descriptors/svm` (SVM backend)
- future: neighbor acceleration (`kiddo`) behind dedicated feature flags

## Fixture plan
- Use `serde` + `serde_json` for test fixtures in Rust.
- Maintain golden fixtures derived from Python reference runs.
- Store fixtures under `rust/fixtures/` once source parity runs are available.
