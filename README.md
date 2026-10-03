# RF1-SRA social reward, context, and closeness — current v4 launch

The current analysis reconstructs QC-qualified samples for **Shared Reward–Trust**
and **Shared Reward–Trust–Social/monetary Doors**, preserving distinct partner maps.
UGR and decision-phase maps are not inputs or eligibility requirements. See the
[revised analysis plan](docs/REVISED_ANALYSIS_PLAN.md) for exact contrasts and tests.

Primary QC uses tSNR, coverage, and mean FD; the historical four-metric rule is a
sensitivity analysis. The primary partner-pair validation roster targets exactly
**50 QC-qualified participants**, supplementing surviving original holdouts only
with new participants before any of their voxel data is read. All original holdouts,
including QC failures, remain barred from training. Other analyses use the eligible
subset of that same roster. No holdout-scoring command is implemented.

On Linux2, from this repository with the existing analysis Python environment:

```bash
git pull --ff-only
bash code/run_revised.sh --dry-run
bash code/run_revised.sh
```

The second command checks the fresh inventory and proposed reconstruction without
saving membership or reading voxel values. The third locks samples and reruns all
revised development models. It requires the original private split and fold files;
never delete them or move development participants into validation. You can use
`--sample-only` to lock/review counts before the full run. Set `PYTHON` if necessary.

Read `results/revised/samples/cohort_summary.tsv`, then
`reports/revised/<policy>/<cohort>/REPORT.md`. Commit public `results/`, `reports/`,
and `provenance/` after successful completion; keep all `work/` outputs private.
The revised command does not overwrite the completed v3 results.

The prior N>=250 / N>=200 development gate belongs to v3 and was superseded by the
approved QC-qualified reconstruction. The approximate primary sample from the posted
QC review is 184 development + 50 validation; only the fresh Linux2 inventory can
establish final counts. Validation remains unscored during development.

---

# Archived v3 pilot documentation

The instructions below describe the completed historical pilot. Use the v4 launcher
above for the revised question; `run_pilot.sh` intentionally retains v3 behavior.

# RF1-SRA trans-task social-reward neural signature

Estimate common and task-specific social-versus-nonsocial reward information across
**Shared Reward, Trust, and Social/monetary Doors** in the same participants.
Shared Reward is not the privileged derivation task. UGR is a social-valuation and
broader decision/context **boundary probe**, never a fourth reward training task.
Shared Reward now uses the verified **RF1 full-trial activation outputs in
`sharedreward-aging`**, including approved one-run L1 passthrough. Its estimates
span decision onset through outcome offset. A separate **Trust/Doors-only 2×2**
decision/context signature family is secondary; Shared Reward decision-phase and
unverified neutral probes are explicitly unavailable.

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
   evidence, and seven-repository provenance; inventory all five task implementations.
2. Establish the ses-01 multitask-complete cohort and enforce its minimum size.
3. Create or hash-check the locked N=50 split, then deterministic participant folds.
4. Build one development-only mask from common reward-paradigm coverage.
5. Fit three task-specific reward classifiers and produce the full 3×3 cross-decoding
   matrix. All tested participants are absent from training in every paradigm.
6. Train on two reward paradigms and test the unseen third in unseen participants,
   repeated for all three held-out paradigms: the primary task-general test.
7. Fit common out-of-fold reward models for UGR, cross-family,
   Trust decomposition, and run-reliability probes.
8. Run the secondary Trust/Doors decision/context task models, 2×2 matrix, common models,
   bidirectional reward/context cross-application, and spatial similarities.
9. After all CV predictions are saved, fit final development-only common and
   task-specific models; generate aggregate tables, maps, figures, and reports.

No preprocessing, TEDANA, FEAT, first-/second-level modeling, or single-trial
estimation is run. Fixed LinearSVC settings and per-map spatial mean centering
are preserved. There is no tuning or algorithm selection.

## Sources and overrides

Default sibling root: `/ZPOOL/data/projects`. All six source repositories are read-only.

| Repository | Inputs |
| --- | --- |
| `rf1-sra-linux2` | Canonical BIDS, baseline participants.tsv, JSON metadata, source-exclusion policy and descriptive QC |
| `rf1-sra-sharedreward` | Existing harmonized RF1 BOLD/grid resources and provenance only |
| `sharedreward-aging` | Verified RF1-only full-trial activation COPEs 1–6: retained-run L2 or one-run L1 passthrough |
| `rf1-sra-trust` | Model 1, sm-5, L2 outcome and choice contrasts |
| `rf1-sra-socdoors` | Separate socialdoors/doors L1 run 1, cope4 outcome and cope3 decision; never their combined L2 |
| `rf1-sra-ugr` | Model 3, sm-5, L2 offer modulation and condition activity |

Configure `config/paths.yaml` or `--config-dir`. Environment overrides:
`PROJECTS_ROOT`, `RF1_LINUX2_ROOT`, `RF1_SHAREDREWARD_ROOT`, `RF1_TRUST_ROOT`,
`RF1_SOCDOORS_ROOT`, `RF1_UGR_ROOT`, `RF1_AGING_ROOT`, `BIDS_ROOT`, `FMRIPREP_ROOT`,
`SOURCEDATA_EXCLUSIONS_ROOT`, `TEMPLATEFLOW_HOME`. All derived outputs remain in
this checkout, disjoint from protected source paths; output symlinks and hard links
are guarded. Upstream code is never executed and sibling repositories are not pulled.

The exclusion root must be readable; absence is not zero exclusions. Only
`Smith-SRA-<ID>` directory existence is inspected, never private contents/reasons.
QC flags are described without inventing new motion or behavioral thresholds.
The retained Shared Reward runs come from the verified frozen aging audit; its
ratings-eligibility flags are not used to filter this neural-signature cohort.

`config/paths.yaml: aging_audit` explicitly pins
`logs/records/full-analysis-20260930-203949-441196`. The adapter checks its final
computational gate, manifest/contrast hashes, RF1 activation candidate identities,
saved model designs, input fingerprints, and headers. No latest-directory guessing,
PPI inputs, ds003745 participants, or unretained runs are allowed. An updated audit
requires an explicit path change to the complete matching record.

Run reliability uses only development participants with two retained runs. The
primary models use all multitask-complete participants, including verified one-run
Shared Reward outputs. See `results/aggregate/analysis_availability.tsv` and
`provenance/sharedreward_source.json` for availability and source provenance.

## Restart and holdout migration

Rerun the same launcher after resolving a failure. The split and matching folds
are reused and checked; modeling outputs are recomputed. A complete
`provenance/run_status.json` certifies successful completion. Interrupted outputs
may be partial/stale and must not be published. Writes are atomic per file; a
process lock prevents concurrent launchers.

Securely back up `work/splits/subject_split_v1.tsv` with its JSON hash. Missing
membership, hash mismatch, or cohort drift stops processing. A legacy split without
`cohort_definition: multitask_complete_v3_aging_fulltrial` is **not** accepted as multitask validation,
even if its IDs happen to match. The dangerous `--regenerate-split` override archives
old membership and metadata locally and warns; it cannot undo previous exposure.
No existing split is silently replaced.

## Outputs and privacy

| Location | Contents / sharing policy |
| --- | --- |
| `work/inventory/`, `work/splits/` | Private inventory and membership; ignored |
| `work/predictions/`, `work/diagnostics/` | Private OOF scores, model membership, norms, resampling records; ignored |
| `results/aggregate/task_overlap.tsv` | Exact task-intersection Ns |
| `results/aggregate/*cross_decoding.tsv` | Reward 3×3 and secondary decision 2×2 cells, exact CIs, margins |
| `results/aggregate/reward_leave_one_paradigm_out.tsv` | Three primary unseen-paradigm tests |
| `results/aggregate/cross_family_decoding.tsv` | Reward → decision and decision → reward, OOF |
| `results/aggregate/common_signature_probes.tsv` | UGR and Trust decomposition probes |
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
