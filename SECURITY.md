# Security Policy

## Supported versions

This repository is under active development. Security fixes are applied to the current `main` branch.

## Reporting a vulnerability

Please open a private security report through GitHub Security Advisories for this repository.

If private reporting is not available, open a regular issue with minimal exploit detail and request a private follow-up.

## Response expectations

- Initial acknowledgement target: within 7 days.
- Triage target: within 14 days.
- Fix timeline depends on severity and release schedule.

## Supply chain checks

From `D:\FuzzyRough\rust`, run:

```bash
cargo audit
cargo deny check
```
