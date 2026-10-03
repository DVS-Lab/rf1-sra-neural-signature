# Revised analysis v4: QC, social reward, context, and closeness

The completed v3 pilot is preserved at `55857ec`; its QC review is at `ee16a9b`.
The user authorized reconstructing the samples, removing UGR and decision-phase
analyses, separating partner conditions, and topping up the primary validation
sample only from new, previously unmodeled participants. This supersedes the v3
N>=250 cohort / N>=200 development requirement. It does not authorize scoring
validation images or moving any original holdout into training.

## Samples and QC

Primary QC uses the three named metrics: low tSNR, low brain coverage, and high
mean FD. The four-metric historical policy, additionally excluding high TEDANA
rejected-component counts, is a sensitivity analysis. Read the canonical
`rf1-sra-linux2/qc/run_qc.tsv` flags; do not recompute thresholds after exclusions.
The source threshold policy is one-sided 1.5-IQR by paradigm, with both Doors tasks
pooled for threshold estimation. Missing selected QC is not a pass.

Construct a fresh header-only inventory for Shared Reward, Trust, Social Doors,
and monetary Doors. Require only the contrast outputs used by v4: SR COPE1-6,
Trust COPE4-9, and both Doors COPE1,2,4. UGR and decision-phase maps no longer gate
eligibility. Original subject-exclusion directories remain authoritative.
Full design provenance, image geometry, and the frozen aging audit remain checked.
The `rf1-sra-sharedreward` repository remains a read-only provenance dependency:
aging's saved fingerprints reference harmonized RF1 input images there. All Shared
Reward analysis COPEs still come from `sharedreward-aging`; restoring this dependency
does not switch to the original repository's FEAT models.

| Sample | Required task implementations | Analyses |
| --- | --- | --- |
| Primary partner pair | Shared Reward + Trust | Partner identity, social reward, social context, three closeness contrasts, valence control |
| Three reward paradigms | Shared Reward + Trust + Social Doors + monetary Doors | Social reward/context task transfer, LOPO, common signatures |

A task is rejected if any run contributing to its existing subject-level map
fails the selected QC policy. SR uses only the one/two runs retained by its frozen
aging audit. Trust's current L2 uses both runs. No run is removed from an existing
L2 average without changing that average; this revision does not recombine runs
or write upstream FEAT outputs.

## Validation roster and exposure protection

1. Require the original private N=50 split, original development folds, and their
   matching SHA-256 provenance. Never redraw this original split.
2. Retain every original holdout that passes primary QC and the fresh partner-pair
   inventory. Keep *all* original 50 barred from training, including QC failures.
3. Before any new participant voxel access, supplement to exactly 50 qualified
   partner-pair validation participants from candidates absent from the original
   split. Uniformly sample without replacement from sorted candidates using seed
   20260928. This preserves existing assignments; it does not claim that the new
   combined roster has the original age/FlipAngle stratification.
4. If too few new candidates qualify, stop before voxel access. Never move an
   exposed original development participant into validation to fill vacancies.
5. Save and hash the entire revised sample manifest. New participants outside that
   validation roster may enter development if QC-qualified. Preserve original
   development folds; assign new development participants deterministically from
   SHA256(seed:subject) modulo five. Use those same folds across tasks/policies.
6. Doors and four-metric sensitivity analyses use the eligible subset of this same
   protected roster. They may have fewer than 50 qualified validation participants;
   do not top up each analysis separately. None is scored in this launcher.
7. On rerun, source-QC/configuration/eligibility drift or missing private membership
   stops execution rather than overwriting the lock. Orphaned revised model outputs
   without a sample manifest also stop execution because exposure is uncertain.

The posted three-metric review suggests 176 surviving original development and
38 original validation participants for the partner pair, plus 20 new candidates.
Reserving 12 of those new candidates would yield approximately **184 development
and 50 validation**. These are provisional counts; the fresh inventory can recover
participants previously blocked by unrelated outputs. The three-paradigm sample's
validation N depends on which protected participants also have both Doors tasks.

A technical gate requires at least ten development participants and all five folds
for each requested scope. This is only a computational check, not a power claim.
The primary validation roster must contain exactly 50 QC-qualified people.

## Exact representations

Let positive maps be SR `(C_rew[2], F_rew[4], S_rew[6])` and Trust
`(C_rec[5], F_rec[7], S_rec[9])`. Negative maps are SR
`(C_pun[1], F_pun[3], S_pun[5])` and Trust `(C_def[4], F_def[6], S_def[8])`.
Order is computer, friend, stranger. Let R = positive - negative and
M = (positive + negative)/2, separately for each partner.

| Family | Positive/first map | Negative/second map or other classes |
| --- | --- | --- |
| Partner reward identity | R_C, R_F, R_S as three distinct classes | Labels 0=computer, 1=friend, 2=stranger; one map per class/person/task |
| Social reward | (R_F+R_S)/2 | R_C |
| Social context | (M_F+M_S)/2 | M_C |
| Positive-outcome closeness | F_rew / F_rec | S_rew / S_rec |
| Negative-outcome closeness | F_pun / F_def | S_pun / S_def |
| Reward-modulation closeness | R_F | R_S |
| Valence positive control | (positive_C+positive_F+positive_S)/3 | (negative_C+negative_F+negative_S)/3 |

For the three-paradigm sample, Doors social reward compares Social Doors
`win-loss[4]` with monetary Doors `win-loss[4]`. Doors social context compares
`(win[1]+loss[2])/2` between those tasks. COPE3 decision maps are not loaded.
SR remains a full-trial model; Trust/Doors contrasts are outcome-phase. The
context average is an operational contrast, not proof of a pure social process.

## Models and tests

Use the existing fixed LinearSVC parameters and per-map spatial mean centering;
no hyperparameter tuning, unit normalization, PCA, or outcome-selected masks.
Build a separate 95%-coverage development-only whole-brain mask for each QC policy
and cohort using its required tasks. Match the public TemplateFlow whole-brain
mask when available. Preserve source headers only after removing private fields.
All sample locks are established before these voxel operations.

For each of the seven partner-pair families, fit two task-specific models and a
pooled SR/Trust model in each participant fold. Test both task-specific models on
both tasks (2x2) and the common model on both tasks: six cells per family, 42 total.
For each of the two three-paradigm families, fit the 3x3 task matrix, three
leave-one-paradigm-out models, and the common model tested on each task: 15 cells
per family, 30 total. Every test participant is absent from training in all tasks.
LOPO additionally excludes the target task entirely.

Repeated-run reliability applies each common binary model out of fold to SR and
Trust run-level maps. SR uses only participants with two retained QC-passing runs;
Trust uses both contributing runs. Report N, Pearson/Spearman correlations and
ICC(3,1) for the paired score margins. This does not fit on run 1 and score run 2.

Fit final task-specific and common development models after obtaining all OOF
predictions. Save weights, intercepts, labels, training counts, iterations, and
hashes. Final development models are not used to estimate development accuracy or
score the validation set. Classifier weights are not psychological localization.

## Inference and reporting

Binary outcomes remain participant-level paired ranking: score(first map) minus
score(second map), with ties counted incorrect. Report the original descriptive
binomial accuracy/CI, bootstrap mean-margin interval, and Holm-adjusted nominal
binary p values within each family/cohort/policy. The correction is not across
the entire project and does not remove dependence induced by overlapping CV fits.

Three-class outcomes are mean accuracy over three maps within each participant.
Bootstrap participants for accuracy intervals; do not count the 3N maps as
independent participants or run a binomial test with denominator 3N. Publish the
confusion matrix, N participants, 1/3 chance reference, and average true-class
margin. Binary and three-class accuracies have different chance levels.

All revised performance is exploratory development. The closely related positive,
negative, and difference closeness comparisons and the two QC policies must not
be treated as independent replications. Preregister final validation contrasts,
metrics, and multiplicity before a separately implemented validation evaluation.
No permutation-refitting test or validation-scoring command is added in this revision.

## Launch, storage, and recovery

```bash
bash code/run_revised.sh --dry-run
bash code/run_revised.sh
```

Optional `--sample-only` saves and locks all reconstructed samples and exits before
voxel access. A later full launch verifies and reuses those assignments. There is
no regeneration flag. Restore missing private locks from secure backup rather
than deleting provenance to rerun. Back up original `work/splits/` together with
`work/revised/samples/` and both generations of matching public provenance.

The revised launcher has its own config `config/revised_analysis.yaml` and outputs:

- Private: `work/revised/{inventory,samples,<policy>/<cohort>}/`.
- Public aggregates/maps/figures: `results/revised/<policy>/<cohort>/`.
- Sample counts: `results/revised/samples/cohort_summary.tsv`.
- Reports: `reports/revised/<policy>/<cohort>/REPORT.md`.
- Status/model/sample hashes: `provenance/revised/`.

The old v3 split, folds, reports, figures, maps, and provenance remain untouched.
The same process lock prevents simultaneous v3/v4 launchers. A complete revised
status certifies all four requested policy/cohort scopes completed. On failure,
status records the failed stage without public participant identifiers.

## Implementation validation

Local validation on 2026-10-03: `python -m pytest -q` passed all 61 tests,
including the original v3 end-to-end regression and all four revised policy/cohort
scopes on synthetic data. Tests prohibit holdout voxel reads, preserve original
split/fold hashes, verify deterministic supplementation, reject insufficient unseen
candidates and missing locks after interrupted runs, and exercise the revised
pipeline with UGR and decision outputs removed. Trust and both Doors contrast
contracts also match the checked-out upstream templates. The shell launcher passes
`bash -n` and the staged changes pass `git diff --check`.

This verifies implementation behavior; no real Linux2 reconstruction, model fit,
or validation scoring was performed here. Final sample counts and revised results
must come from the Linux2 launch.
