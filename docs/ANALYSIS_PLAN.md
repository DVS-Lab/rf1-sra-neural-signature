# Trans-task social-reward pilot — architecture v2

## Question and cohort

The primary construct is social-versus-nonsocial reward/outcome processing shared
across **Shared Reward, Trust, and Social/monetary Doors**. Each paradigm has equal
standing as a training and test task. UGR assesses broader social valuation/context;
it is not categorized or trained as another reward paradigm.

Use only ses-01 participants with complete required core activation maps in all
five implementations: sharedreward, trust, socialdoors, doors, ugr. Inventory
records individual availability privately and reports every nonempty combination
of the five tasks (31 exact intersections), plus task-specific missingness. The
intersection includes clean decision/context maps and Shared Reward run maps.
Missing secondary paradigms no longer permit inclusion in a single-task cohort.

The explicitly chosen launch gate is **N≥250 multitask-complete**, so reserving
exactly 50 leaves at least 200 development participants. Below that, STOP before
split creation and report exact overlap Ns; never fall back to partial eligibility.
The synthetic test config deliberately uses a lower gate, labeled as a fixture.
Changing the production gate requires an explicit pre-analysis configuration
revision; its value is recorded with the split and report.

Canonical source exclusion is based solely on directory existence. No new motion,
behavioral, or questionnaire thresholds are introduced. Non-mandatory imaging
flags are descriptive. No private raw behavior is read. Actual completed designs
must match current EV ordering, contrast weights, smoothing, and fixed-effects
input contracts. A scientific contradiction in any core design stops processing.
Specific MNI identity must be established; generic MNI NIfTI codes alone do not
identify a template variant. Missing mechanical evidence makes that task unavailable.

## Locked split and folds

Before any voxel access, select N=50 from the multitask-complete cohort with seed
20260928. Try age quintile × Shared Reward FlipAngle strata, quartiles, tertiles,
then FlipAngle alone; record all fallbacks. Never infer missing age/FlipAngle.
Unknown and discrepant acquisition values are explicit categories. Sex is described,
not used as a class target. These demographics never enter the neural classifier.

Persist and hash membership. Reject legacy single-task split provenance, missing
private membership, changed hashes, and cohort drift. No automatic redraw.
Create five deterministic participant folds within development, using feasible
age/FlipAngle strata; explicitly report participant KFold fallback when necessary.
The **same folds govern every model, paradigm, and contrast family**. All maps and
both class labels belonging to a participant remain together.

## Exact representations

All arithmetic uses existing COPEs in memory. L2 k means `copek.feat/stats/cope1.nii.gz`.

| Family / paradigm | SOCIAL (+1) | NONSOCIAL (−1) |
| --- | --- | --- |
| Reward: Shared Reward L2 | ((4−3)+(6−5))/2 | 2−1 |
| Reward: Trust L2 | ((7−6)+(9−8))/2 | 5−4 |
| Reward: Social/monetary Doors | socialdoors L1 run-1 cope4 | doors L1 run-1 cope4 |
| Secondary decision: Shared Reward L2 | (27+28)/2 | 29 |
| Secondary decision: Trust L2 | (2+3)/2, friend/stranger choice | 1, computer choice |
| Secondary decision: Social/monetary Doors | socialdoors L1 run-1 cope3 | doors L1 run-1 cope3 |
| UGR offer/fairness probe L2 | 14 | 13 |
| UGR broader social context probe L2 | (5+7)/2 | (1+3)/2 |
| Shared Reward neutral probe L2 | (8+9)/2 | 7 |
| Shared Reward reliability L1, each run | ((4−3)+(6−5))/2 | 2−1 |

Trust descriptive outcome decompositions are friend vs computer (7−6 vs 5−4),
stranger vs computer (9−8 vs 5−4), and friend vs stranger (7−6 vs 9−8).
UGR copes 13/14 sum endowment-specific offer pmods; upstream offers are demeaned
within sociality × endowment × run. UGR constants cover broad epochs and are **not
pure isolated decision contrasts**. UGR never enters reward/context training.

## Common mask, geometry, normalization

Retain the Shared Reward reference grid as a spatial convention, not a privileged
learning target. For each development participant, intersect required FEAT masks
across Shared Reward, Trust, socialdoors, and doors, with nearest-neighbor grid
resampling where same-space provenance is verified. Retain voxels covered in at
least 95% of these participant intersections. Intersect with the local whole-brain
TemplateFlow MNI152NLin6Asym res-02 brain mask when reliably available; otherwise
report the development multitask-coverage-only fallback. UGR masks are not used
to select the training feature space. No holdout masks or condition values enter
mask construction. Mask-source resampling is logged separately from statistical maps.

The mask is fixed across all development models and uses no labels or performance.
CV inference is conditional on this development-wide coverage mask. Shared Reward
reference grids must agree; clearly same-MNI-space task maps on other grids are
linearly resampled in analysis copies with each operation logged. No source writes.
Header inspection establishes finite geometry and nonempty files, not finite unread
holdout voxels. A nonfinite/inaccessible required development map stops the revised
pipeline rather than silently producing different cell-specific cohorts after locking.

Construct each representation, vectorize under the same boolean mask in C-order,
and subtract that map's spatial mean. No unit norm, global z-scoring, task scaling,
PCA, supervised feature selection, or performance-driven preprocessing. Save norms.
All task/participant pairs have equal sample weight; residual norm differences may
influence pooled fitting and are reported rather than tuned away.

## Model and evaluation architecture

Use LinearSVC(C=1, dual=True, class_weight=None, max_iter=100000, tol=0.0001,
random_state=20260928). Nonconvergence stops the run. No hyperparameter search or
algorithm comparison. SOCIAL=+1 and NONSOCIAL=−1 in both families.

For each of the five participant folds:

1. **Task-specific reward models / 3×3 matrix:** train one model on each reward
   paradigm using training participants only; apply it to all three paradigms in
   the held-out participants. Diagonal cells test unseen participants within task;
   off-diagonal cells test unseen participants and unseen paradigms.
2. **Primary leave-one-paradigm-out tests:** for each reward paradigm B, train on
   the other two paradigms A/C using only training participants. Score B only in
   held-out participants. Explicit guards reject both participant overlap and B's
   presence in the model's training-task metadata. Three prespecified tests are
   reported separately; do not treat the three margins per participant as independent.
3. **Common reward model:** train on all three reward paradigms in training
   participants. Score held-out participants' reward maps, UGR pmod/constants,
   Shared Reward neutral maps, Trust outcome decompositions, both Shared Reward
   runs, and all three decision/context maps. Its within-paradigm tests assess
   unseen people; they are not presented as unseen-paradigm validation.
4. **Secondary decision/context family:** fit each task-specific context model and
   the complete 3×3 matrix, using the same participant folds. Fit a common context
   model on all three clean context paradigms and apply it to held-out context,
   reward, UGR, and neutral maps. Reward→context and context→reward applications
   are therefore participant-independent in both directions.

After all CV predictions are written, fit final common and task-specific models
on all development participants for both families. The final common reward model
receives six maps per participant (three social/nonsocial pairs), equally weighted.
Save final weights, fold weights, intercepts, training-task metadata, runtime
versions, source SHAs, mask/split hashes, and iteration counts. No final model
is applied to the holdout or used for development performance estimates.

## Statistics and comparison

Each reported cell has one paired margin per participant, score(social) minus
score(nonsocial); the intercept cancels. Forced choice is margin>0. Report N,
correct N, ties (counted incorrect), accuracy, two-sided Clopper–Pearson 95% CI,
two-sided exact binomial p against .50, mean/median margin, and the percentile
95% CI from 10,000 participant bootstrap means (seed 20260928). Never pool rows
across paradigms as independent people. The three LOPO tests are prespecified
primary pilot tests; p values are unadjusted, not a confirmatory multiplicity claim.
Overlapping training sets mean these intervals do not capture full model-fitting
uncertainty. No task result changes the analysis specification.

Report common reward-model OOF Shared Reward run-margin Pearson r, Spearman rho,
ICC(3,1), both run accuracies, and an unlabelled scatter. ICC(3,1) is two-way mixed,
consistency, single measurement: (MS_subject−MS_error)/(MS_subject+(k−1)MS_error),
k=2. Constant vectors yield undefined coefficients. It is not absolute-agreement ICC.

Compute all task/common reward/context map correlations separately for each CV
fold and for final development maps. Report each model's 5×5 fold stability matrix
and off-diagonal mean/min/max. Weight correlations are descriptive, affected by
shared training data and correlated features; they are not psychological localization
or proof of construct identity. Bidirectional cross-family decoding adds evidence
about common versus distinct information but cannot alone prove a selective mechanism.

Reports explicitly address: within-task decoding; all cross-task transfers;
unseen-paradigm LOPO performance; task weight similarity; UGR common-model transfer;
and reward/context similarity versus dissociation. Strong UGR transfer is consistent
with broader social-value information. Weak UGR transfer with successful reward
decoding is consistent with greater reward/outcome specificity, subject to power,
reliability, and measurement differences. Neither outcome prompts tuning.

## Future external adapter

A future reviewed adapter supplies participant ID, positive/negative maps, optional
run, and explicit template/grid metadata. It can reuse `spatial_center`,
`reconstruct`, `center_pair`, guarded `score_pair`, and `reporting.performance`.
The current guard permits only locked development participants. External validation
requires a separately authorized cohort guard and a frozen-model loader using the
saved common reward weight map and its intercept in `provenance/model.json`.
Do not loosen the RF1 holdout guard. The future internal validation can evaluate
all three reward paradigms and UGR in the same 50 unseen participants after freeze
and a specified, ideally preregistered confirmatory protocol.

API references: [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
and [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
Runtime versions are recorded rather than inferred from online documentation.
