# Implementation validation — trans-task architecture v2

**No real Linux2 RF1 analysis or holdout scoring was run.** These checks use only
synthetic images, metadata, source repositories, and participants.

## Results

- Complete suite: **33 passed in 132.88 seconds** (`python3 -m pytest -q`).
- Synthetic end-to-end cohort: 70 complete participants; 50 locked and 20 development.
  The temporary fixture explicitly uses a 55-person launch minimum and 300 bootstrap
  draws for speed. The committed production defaults remain **250 complete / 200
  development minimum**, N=50 holdout, and 10,000 bootstrap draws.
- All 44 inferential cells had exactly 20 held-out participant margins: 18 pairwise
  cells (reward and secondary decision 3×3 matrices), 3 reward LOPO tests, 6 common
  within-paradigm cells, 6 bidirectional cross-family cells, 9 probes, and 2 run cells.
- Generated **64 maps** (mask, 55 CV coefficient maps, 8 final development maps),
  **9 figures**, aggregate tables, private predictions, membership audits, and reports.
- Model-membership audit verifies disjoint training/test participants for every
  model and fold. LOPO training-task metadata omit the held-out paradigm. UGR
  appears in no training model. Final models are fit only after CV predictions save.
- A patched image loader raises on any holdout `get_fdata()` call; the full run passed.
  Source-file content digests matched before and after the complete run.
- Actual shell dry-run passed on the complete fixture without creating a split or
  model output. A deficient fixture stopped with exact intersection Ns before a split.
- The production-size gate was tested at both sides: N=249 stops, N=250 yields
  exactly 200 development and 50 holdout. Single-task fallback and legacy split
  reuse fail intentionally. Existing hash, drift, missing-private-file, reproducibility,
  and dangerous-regeneration safeguards remain tested.
- Exact reward/decision arithmetic, source EV/vector drift, separate Doors L1 maps,
  fixed-effects mean contracts, participant/LOPO leakage guards, and UGR exclusion
  from training passed.
- Common-mask 95% coverage, TemplateFlow fallback/intersection, NIfTI vector order,
  resampling, output header privacy, exact/binomial/bootstrapped summaries,
  ICC(3,1), output schemas, symlink/hard-link protections, and public-output
  participant-ID scans passed.
- Current checked-out upstream L1/L2 templates were reverified, including Trust
  choice COPEs 1–3 and separate Doors decision COPE3. No upstream files changed.
- Shell syntax, Python compilation, configuration parsing, and Git whitespace checks
  passed. Synthetic cross-decoding and similarity figures were visually inspected.

## Environment

Python 3.13.5; numpy 2.1.3; pandas 2.2.3; scipy 1.15.3; scikit-learn 1.6.1;
nibabel 5.3.2; PyYAML 6.0.2; matplotlib 3.10.0; pytest 8.3.4.
Real runs record their own software and repository versions. Previous v1 validation
is retained in Git history and does not establish the revised scientific analysis.

## Linux2 observations still required

The real dry run must establish all five task implementations' completed FEAT
outputs and saved designs, readable referenced MNI BOLD headers, canonical baseline
demographics/BIDS acquisition metadata, the authoritative source-exclusion directory,
and the exact multitask intersection N. The default minimum of 250 is not a claim
that 250 are available. Local TemplateFlow is optional with the documented common
coverage fallback. Failed contracts never trigger source regeneration, new scientific
exclusions, a partial-cohort fallback, holdout scoring, or model tuning.
