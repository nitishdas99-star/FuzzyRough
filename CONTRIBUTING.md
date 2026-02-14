# Contributing

## Development workflow

1. Fork the repository and create a feature branch from `main`.
2. Keep changes focused and include tests.
3. Run formatting, linting, and tests before opening a PR.

## Branch naming

- `feat/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`
- `chore/<short-description>`

## Local checks

From `D:\FuzzyRough\rust`:

```bash
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Optional supply-chain checks:

```bash
cargo audit
cargo deny check
```

## PR checklist

- [ ] Code compiles on stable Rust.
- [ ] `cargo fmt` applied.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes.
- [ ] `cargo test --workspace` passes.
- [ ] README and docs updated if public behavior changed.
- [ ] Changelog entry added under `Unreleased` (when appropriate).
