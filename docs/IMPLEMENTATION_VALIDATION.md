# Implementation validation

Validation used synthetic data only. No Linux2 RF1 participant data, real split,
neural result, or holdout performance was accessed or produced.

## Completed checks

- Full suite: **30 passed** (`python3 -m pytest -q`, 303.43 seconds).
- Final focused rerun after header-privacy and launcher-interpreter fixes:
  **29 passed, 1 deselected** (the already-passed full synthetic pilot; 19.46 seconds).
- End-to-end fixture: 70 synthetic participants, exactly 50 locked holdout and 20
  development; primary and secondary missingness, five fold coefficient maps,
  final development map, mask, five figures, private predictions, aggregate
  summaries, and both reports generated successfully.
- A patched image loader raises if any holdout `get_fdata()` is called. The full
  pipeline passed with that tripwire active.
- Source-file content digests before/after the complete synthetic run matched.
- Dry run through the actual shell launcher created no split or model output.
- Split reuse, reproducibility, hash mismatch, lost private membership, cohort
  drift, infeasible strata, grouped pairs, OOF scoring rejection, and intentional
  holdout fitting/loading rejection passed.
- All exact contrast formulas, EV/vector drift checks, separate Doors L1 paths,
  fixed-effects mean contract, 95% coverage, TemplateFlow fallback/intersection,
  resampling logs, vector ordering/NIfTI reconstruction, output schemas,
  binomial/bootstrap summaries, and ICC(3,1) reference cases passed.
- Output traversal, symlink, hard-link protection, and absence of synthetic
  participant IDs in aggregate/report/provenance outputs passed.
- Public NIfTI header privacy was additionally checked after clearing free-text
  fields and extensions. Figure formatting was visually inspected and a clipped
  label corrected; final-format synthetic reports/figures were regenerated.
- `bash -n code/run_pilot.sh`, Python compilation of `code/` and `tests/`, YAML
  parsing, and `git diff --cached --check` passed.
- Current checked-out upstream L1 and L2 templates passed the configured semantic
  contract checks. All five inspected upstream clones remained Git-clean.
- Git ignore rules were checked for the split, predictions, and participant inventory.

The initial synthetic acquisition fixture incorrectly put a preprocessed image in
BIDS; it was corrected and acquisition discovery now ignores derivative filenames.
The launcher test explicitly uses the running test interpreter so unrelated system
Python installations do not determine its dependency availability.

## Validation environment

Python 3.13.5; numpy 2.1.3; pandas 2.2.3; scipy 1.15.3; scikit-learn 1.6.1;
nibabel 5.3.2; PyYAML 6.0.2; matplotlib 3.10.0; pytest 8.3.4.

Every real Linux2 model records its own runtime versions. These synthetic results
do not establish actual RF1 data availability, scanner-space correctness, or model
performance. Run the dry run on Linux2 before the real launcher.

## Linux2 assumptions still requiring observation

1. The five sibling repositories and their expected FEAT output families exist.
2. Completed FEAT `design.fsf`/`design.con` files establish the configured model;
   referenced canonical/harmonized MNI BOLD headers remain readable.
3. Canonical BIDS baseline participants.tsv and acquisition metadata are accessible.
   Missing/discrepant demographic/acquisition values remain explicit rather than inferred.
4. The authoritative private source-exclusion directory is readable. No contents
   or individual exclusion reasons are required.
5. TemplateFlow's exact whole-brain mask is optional; its absence produces the
   documented development FEAT-coverage fallback.

No source regeneration, new exclusions, holdout scoring, or method tuning is
performed to resolve failed contracts.
