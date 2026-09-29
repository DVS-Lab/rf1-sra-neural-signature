# Implementation review — real RF1 pilot not run

Current upstream templates and scientific definitions were checked at the SHAs
in `provenance/implementation_sources.json`. The complete pipeline is implemented;
Linux2 image availability and actual eligible sample size remain unverified.

- N=50 holdout: selected and locked only on first real launch; no scoring API.
- Primary CV, Social Doors, Trust, neutral/decision controls, UGR, reliability,
  and weight stability: implemented; real results pending Linux2 execution.
- Contrasts and exact arithmetic: `docs/ANALYSIS_PLAN.md` and
  `config/contrast_spec.yaml`.
- Private outputs: ignored `work/`; shareable outputs: `results/`, `provenance/`,
  and `reports/`. No real participant-level information was added.
- Operational assumptions: readable canonical source-exclusion directory,
  participants.tsv/BIDS metadata, complete FEAT outputs with saved design evidence,
  and readable source MNI BOLD headers. No source repositories are modified.

The real launcher replaces this file with concise aggregate results, repository
SHAs, warnings, and key output paths. Synthetic validation is recorded separately
in `docs/IMPLEMENTATION_VALIDATION.md`.
