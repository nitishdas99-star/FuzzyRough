# frlearn_lab

`frlearn_lab` is the synthetic benchmark harness in this workspace. It stress-tests preprocessing, classifier, and novelty pipelines with deterministic synthetic datasets and reports both console summaries and optional JSON output.

## What It Tests

- Preprocessing: `RangeNormaliser`.
- Feature/prototype selection hooks: `FRFS` / `FRPS` adapter paths (currently compatibility placeholders in `src/adapters.rs`).
- Classifiers: `NN`, `FRNN`, plus adapter-routed `FRONEC`/`FROVOCO` compatibility wrappers.
- Descriptors: `NND`, adapter-routed `LNND` compatibility wrapper.
- Metrics:
  - Classification: accuracy, macro-F1, confusion matrix.
  - Novelty: ROC-AUC, TPR@1% FPR, TPR@5% FPR.

## Dataset Reference (A–E)

- A: `overlapping_gaussians` (`DatasetChoice::Overlap`)
  - 3-class overlapping Gaussian structure.
  - Stresses class ambiguity and fuzzy boundary behavior.
- B: `xor_with_distractors` (`DatasetChoice::Xor`)
  - XOR signal in first two dimensions + noisy distractor dimensions.
  - Stresses non-linear separability with irrelevant features.
- C: `redundant_irrelevant` (`DatasetChoice::Redundant`)
  - Informative + redundant + irrelevant feature mix.
  - Stresses feature robustness under noisy high-dimensional inputs.
- D: `prototype_heavy` (`DatasetChoice::PrototypeHeavy`)
  - Multi-cluster class structure with larger prototype load.
  - Stresses neighbour-heavy classification throughput.
- E: `one_class_novelty` (used by novelty suite)
  - Trains on inliers only; tests on inlier/outlier mix.
  - Stresses anomaly scoring and threshold behavior.

## CLI Reference

Run from workspace root:

```bash
cd D:\FuzzyRough\rust
cargo run -p frlearn_lab -- --suite all
```

Available flags (from `--help`):

| flag | default | description |
| --- | --- | --- |
| `--suite <all|classifier|novelty>` | `all` | which benchmark suite to run |
| `--dataset <overlap|xor|redundant|prototype-heavy>` | `overlap` | synthetic dataset for classifier suite |
| `--backend <bruteforce>` | `bruteforce` | neighbour backend |
| `--n-train <usize>` | `1000` | number of training samples |
| `--n-test <usize>` | `1000` | number of test samples |
| `--dims <usize>` | `10` | feature dimensions |
| `--noise <f64>` | `0.25` | dataset noise level |
| `--seed <u64>` | `42` | deterministic seed |
| `--k <usize>` | `5` | neighbour count |
| `--no-frfs` | `false` | disable FRFS adapter stage |
| `--no-frps` | `false` | disable FRPS adapter stage |
| `--out <path>` | unset | write JSON report to file |

Note: `--label-noise` is not currently implemented in this CLI.

### Examples

```bash
# Full run (classifier + novelty) with JSON output
cargo run -p frlearn_lab -- --suite all --seed 11 --dims 12 --n-train 1200 --n-test 800 --noise 0.2 --k 7 --out results.json

# Classifier-only run on XOR data
cargo run -p frlearn_lab -- --suite classifier --dataset xor --n-train 1500 --n-test 1500 --seed 3

# Novelty-only run with FRFS/FRPS disabled
cargo run -p frlearn_lab -- --suite novelty --no-frfs --no-frps --out novelty_report.json
```

## Output Formats

- Console output:
  - Classification table: model, accuracy, macro-F1, timing, train size, effective dims.
  - Novelty table: descriptor metrics and timing.
- JSON output:
  - Enabled with `--out <path>`.
  - File is written exactly to the given path (for example `results.json` in current directory).
  - Serialized via `JsonReport` (`src/report.rs`), including model reports, confusion matrices, and novelty metrics.

## Suite Semantics

- `all`: runs `run_classifier_suite` and `run_novelty_suite` in one invocation.
- `classifier`: runs synthetic classification benchmarks only.
- `novelty`: runs one-class novelty benchmarks only.
- Backend consistency:
  - Current runtime backend path is brute force.
  - `kdtree` / `balltree` are feature hooks; enable only when corresponding workspace support is implemented.

## How To Add a New Benchmark Suite

1. Add a new enum variant in `src/runner.rs` (`SuiteChoice`) and wire clap parsing via `ValueEnum` usage in `src/main.rs` if needed.
2. Implement a dedicated runner function (similar to `run_classifier_suite` / `run_novelty_suite`) in `src/runner.rs`.
3. Extend report structs in `src/report.rs` if the suite needs new metrics or output fields.
4. Add adapter methods in `src/adapters.rs` for new algorithm categories or facade mismatches.
5. Add synthetic dataset builders in `src/datasets.rs` when new stress profiles are required.
6. Add tests:
   - smoke-level behavior in `tests/smoke.rs`
   - metric/unit tests in `src/metrics.rs` or dedicated tests.
7. Validate:
   - `cargo fmt`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo test --workspace`
