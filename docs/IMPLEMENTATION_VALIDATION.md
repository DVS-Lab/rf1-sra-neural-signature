# Implementation validation — aging full-trial adapter v3

**No real Linux2 RF1 voxel analysis or holdout scoring was run here.** Real-source
inspection used the tracked aging audit and code at
`323902ca9dd11871d04219a2f89c40ddb3bfa5d3`; computational tests use synthetic images.

## Validation scope

- Final clean suite: **46 passed in 266.21 seconds** (`python3 -m pytest -q`).
  Shell syntax, Python compilation, production config parsing and Git whitespace
  checks also passed. Earlier intermediate fixture/code revisions are superseded
  by this final clean run.
- The integration fixture has 70 multitask participants, mixing two-run fixed effects
  and single-run Shared Reward L1 passthrough. Exactly 50 are held out and 20 develop.
  Only the temporary fixture lowers the cohort gate to 55 and bootstrap draws to 300.
  Production remains N>=250, N=50 holdout, and 10,000 bootstrap draws.
- The reward 3x3 matrix, three LOPO cells, common reward model, UGR and Trust probes
  use all development participants. The decision matrix is Trust/Doors 2x2. Common
  decision models use these two tasks; cross-family scoring uses the available
  target family. Shared Reward decision and neutral analyses are explicitly unavailable.
- The full run produces 35 inferential cells: 13 pairwise, 3 LOPO, 5 common,
  5 cross-family, 7 probes, and 2 reliability cells. All non-reliability cells have
  N=20; reliability uses the two-retained-run development subset and reports its N.
- Expected outputs: 58 NIfTI maps (50 fold models, 7 final models, 1 mask), 9 figures,
  availability/source provenance, aggregate reports, and private prediction/membership
  tables. Synthetic decision heatmap layout was visually checked.
- A patched image loader fails on any holdout get_fdata call. Source-file content
  digests are compared before/after; participant IDs are scanned out of public output.
  Model audits establish disjoint training/test participants and LOPO task exclusion.
- Actual shell dry-run checks write no outputs or split. Insufficient cohorts stop
  before split. Production N=249/N=250 gate checks, locked split hash/drift/reuse,
  and holdout fitting/scoring guards remain covered. Earlier cohort contracts cannot
  be silently reused as v3, even with matching participants.
- Source adapter tests cover actual aging path structure, both single-run choices,
  fixed-effects means, ignored external/PPI rows, wrong participant/contrast/strategy,
  missing provenance, stale image input fingerprints, changed frozen manifests,
  failed upstream audits, and missing unrequired neutral estimates. Provenance cannot
  hash voxel files during the header-only inventory.
- Trust/Doors/UGR arithmetic, participant folds, fixed mask, resampling, image-header
  privacy, exact/binomial/bootstrap inference, ICC(3,1), and output path protections
  remain covered. No new algorithm tuning, source writes, or upstream jobs are used.

## Environment

Python 3.13.5; numpy 2.1.3; pandas 2.2.3; scipy 1.15.3; scikit-learn 1.6.1;
nibabel 5.3.2; PyYAML 6.0.2; matplotlib 3.10.0; pytest 8.3.4.
Real runs record their own software and repository versions.

## Linux2 check still required

Pull main and run `bash code/run_pilot.sh --dry-run`. This must establish current
file availability, saved input/design provenance, headers, demographics and the
five-task intersection. The frozen audit's 346 RF1 activation subjects are not a
claim that 250 are multitask-complete. No holdout is selected by dry-run.
