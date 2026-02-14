use clap::Parser;
use frlearn_lab::report::JsonReport;
use frlearn_lab::runner::{BackendChoice, DatasetChoice, LabConfig, SuiteChoice};

#[derive(Parser, Debug)]
#[command(
    name = "frlearn-lab",
    version,
    about = "Synthetic benchmark harness for fuzzy-rough Rust crate"
)]
struct Cli {
    /// Which suite to run.
    #[arg(long, value_enum, default_value_t = SuiteChoice::All)]
    suite: SuiteChoice,

    /// Dataset for classifier suite.
    #[arg(long, value_enum, default_value_t = DatasetChoice::Overlap)]
    dataset: DatasetChoice,

    /// Neighbour search backend.
    #[arg(long, value_enum, default_value_t = BackendChoice::Bruteforce)]
    backend: BackendChoice,

    /// Number of training samples.
    #[arg(long, default_value_t = 1000)]
    n_train: usize,

    /// Number of test samples.
    #[arg(long, default_value_t = 1000)]
    n_test: usize,

    /// Feature dimensionality.
    #[arg(long, default_value_t = 10)]
    dims: usize,

    /// Gaussian noise level injected into features (dataset-dependent).
    #[arg(long, default_value_t = 0.25)]
    noise: f64,

    /// Random seed for deterministic runs.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Neighbour count (k).
    #[arg(long, default_value_t = 5)]
    k: usize,

    /// Disable feature selection (FRFS).
    #[arg(long, default_value_t = false)]
    no_frfs: bool,

    /// Disable prototype/instance selection (FRPS).
    #[arg(long, default_value_t = false)]
    no_frps: bool,

    /// Optional output JSON file path.
    #[arg(long)]
    out: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    let cfg = LabConfig {
        suite: cli.suite,
        dataset: cli.dataset,
        backend: cli.backend,
        n_train: cli.n_train,
        n_test: cli.n_test,
        dims: cli.dims,
        noise: cli.noise,
        seed: cli.seed,
        k: cli.k,
        use_frfs: !cli.no_frfs,
        use_frps: !cli.no_frps,
    };

    let suite_report = frlearn_lab::runner::run(cfg);
    frlearn_lab::report::print_suite_report(&suite_report);

    if let Some(path) = cli.out {
        let json = JsonReport::from_suite(&suite_report);
        if let Err(e) = std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()) {
            eprintln!("Failed to write JSON report to {}: {}", path, e);
        } else {
            println!("\nWrote JSON report to {}", path);
        }
    }
}
