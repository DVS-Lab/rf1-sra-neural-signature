# RF1-SRA trans-task social-reward neural signature

Estimate common and task-specific social-versus-nonsocial reward information across
**Shared Reward, Trust, and Social/monetary Doors** in the same participants.
Shared Reward is not the privileged derivation task. UGR is a social-valuation and
broader decision/context **boundary probe**, never a fourth reward training task.
A separate decision/context signature family is explicitly secondary.

Exactly **50 multitask-complete participants** form a locked internal multitask
validation set. No holdout voxel loading, pattern expression, or scoring is
implemented. No real Linux2 analysis was run during implementation.

## Linux2 launch

Work is committed directly to `main` in this repository. After pulling:

```bash
cd /ZPOOL/data/projects/rf1-sra-neural-signature
git switch main
git pull --ff-only
python3 -m pip install -r requirements.txt
python3 -m pytest -q
bash code/run_pilot.sh --dry-run
bash code/run_pilot.sh
```

Use Python 3.10+; set `PYTHON=/path/to/python` for the launcher if needed. Tests
use synthetic data and require neither `/ZPOOL` nor FSL. The full integration
fixture explicitly lowers the cohort gate in its temporary config; the production
minimum remains **250 complete participants**, leaving at least **200 development**
after reserving 50. This minimum was explicitly chosen before real data analysis.

Dry run prints exact counts for all 31 nonempty task intersections and stops if
the complete cohort is below 250. A real launch also saves these aggregate counts
before stopping. There is no fallback to Shared Reward or another partial cohort.
Missing or ambiguous core maps cannot silently reduce an individual matrix cell's
sample after the split; development voxel failures stop the run for mechanical review.

If a task has zero complete participants, the inventory prints the unavailable-input
reasons and expected FEAT layouts with directory counts. Real launches save
`results/aggregate/feat_path_summary.tsv`; dry runs print diagnostics without writing.
Shared Reward diagnosis also checks the canonical Linux2 `derivatives/fsl` tree for
alternate activation layouts. These counts establish directory existence only;
alternate layouts are never selected automatically. See
[the missing-input investigation](docs/MISSING_SHAREDREWARD_INPUTS.md).

## What the one command runs

1. Verify paths, current activation templates, saved FEAT designs, specific MNI-space
   evidence, and six-repository provenance; inventory all five task implementations.
2. Establish the ses-01 multitask-complete cohort and enforce its minimum size.
3. Create or hash-check the locked N=50 split, then deterministic participant folds.
4. Build one development-only mask from common reward-paradigm coverage.
5. Fit three task-specific reward classifiers and produce the full 3×3 cross-decoding
   matrix. All tested participants are absent from training in every paradigm.
6. Train on two reward paradigms and test the unseen third in unseen participants,
   repeated for all three held-out paradigms: the primary task-general test.
7. Fit common out-of-fold reward models for UGR, neutral-context, cross-family,
   Trust decomposition, and run-reliability probes.
8. Run the secondary decision/context task models, 3×3 matrix, common models,
   bidirectional reward/context cross-application, and spatial similarities.
9. After all CV predictions are saved, fit final development-only common and
   task-specific models; generate aggregate tables, maps, figures, and reports.

No preprocessing, TEDANA, FEAT, first-/second-level modeling, or single-trial
estimation is run. Fixed LinearSVC settings and per-map spatial mean centering
are preserved. There is no tuning or algorithm selection.

## Sources and overrides

Default sibling root: `/ZPOOL/data/projects`. All five repositories are read-only.

| Repository | Inputs |
| --- | --- |
| `rf1-sra-linux2` | Canonical BIDS, baseline participants.tsv, JSON metadata, source-exclusion policy and descriptive QC |
| `rf1-sra-sharedreward` | Model 1, smTo-6, L2 activation and L1 runs 1/2 |
| `rf1-sra-trust` | Model 1, sm-5, L2 outcome and choice contrasts |
| `rf1-sra-socdoors` | Separate socialdoors/doors L1 run 1, cope4 outcome and cope3 decision; never their combined L2 |
| `rf1-sra-ugr` | Model 3, sm-5, L2 offer modulation and condition activity |

Configure `config/paths.yaml` or `--config-dir`. Environment overrides:
`PROJECTS_ROOT`, `RF1_LINUX2_ROOT`, `RF1_SHAREDREWARD_ROOT`, `RF1_TRUST_ROOT`,
`RF1_SOCDOORS_ROOT`, `RF1_UGR_ROOT`, `BIDS_ROOT`, `FMRIPREP_ROOT`,
`SOURCEDATA_EXCLUSIONS_ROOT`, `TEMPLATEFLOW_HOME`. All derived outputs remain in
this checkout, disjoint from protected source paths; output symlinks and hard links
are guarded. Upstream code is never executed and sibling repositories are not pulled.

The exclusion root must be readable; absence is not zero exclusions. Only
`Smith-SRA-<ID>` directory existence is inspected, never private contents/reasons.
QC flags are described without inventing new motion or behavioral thresholds.

## Restart and holdout migration

Rerun the same launcher after resolving a failure. The split and matching folds
are reused and checked; modeling outputs are recomputed. A complete
`provenance/run_status.json` certifies successful completion. Interrupted outputs
may be partial/stale and must not be published. Writes are atomic per file; a
process lock prevents concurrent launchers.

Securely back up `work/splits/subject_split_v1.tsv` with its JSON hash. Missing
membership, hash mismatch, or cohort drift stops processing. A legacy split without
`cohort_definition: multitask_complete_v2` is **not** accepted as multitask validation,
even if its IDs happen to match. The dangerous `--regenerate-split` override archives
old membership and metadata locally and warns; it cannot undo previous exposure.
No existing split is silently replaced.

## Outputs and privacy

| Location | Contents / sharing policy |
| --- | --- |
| `work/inventory/`, `work/splits/` | Private inventory and membership; ignored |
| `work/predictions/`, `work/diagnostics/` | Private OOF scores, model membership, norms, resampling records; ignored |
| `results/aggregate/task_overlap.tsv` | Exact task-intersection Ns |
| `results/aggregate/*cross_decoding.tsv` | Reward and secondary decision 3×3 cells, exact CIs, margins |
| `results/aggregate/reward_leave_one_paradigm_out.tsv` | Three primary unseen-paradigm tests |
| `results/aggregate/cross_family_decoding.tsv` | Reward → decision and decision → reward, OOF |
| `results/aggregate/common_signature_probes.tsv` | UGR, neutral and Trust decomposition probes |
| `results/aggregate/signature_spatial_similarity.tsv` | Task/common reward/context map correlations, by fold and final DEV |
| `results/maps/common_reward_signature_DEV_weights.nii.gz` | Final pooled development reward signature |
| `results/maps/common_decision_signature_DEV_weights.nii.gz` | Secondary pooled development context signature |
| `results/maps/*fold-*_weights.nii.gz` | All task, LOPO and common CV coefficients |
| `results/figures/cross_decoding_heatmap.png` | Central reward cross-decoding figure |
| `provenance/`, `reports/` | Aggregate descriptions, model settings/intercepts/SHAs, concise review and full report |

Only aggregate results/maps/figures/provenance are shareable. Public NIfTI headers
have participant-bearing free text/extensions removed. Never force-add participant
products or individual source-exclusion information to Git.

See [analysis plan](docs/ANALYSIS_PLAN.md), [holdout policy](docs/HOLDOUT_POLICY.md),
[upstream audit](docs/UPSTREAM_CONTRACT_AUDIT.md), and
[implementation validation](docs/IMPLEMENTATION_VALIDATION.md). The plan documents
future external adapters; no external dataset or holdout scoring is implemented.
