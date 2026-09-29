# Development pilot analysis plan

## Cohort and preprocessing boundaries

Use ses-01 only. Eligibility requires canonical non-source-excluded BIDS membership,
both Shared Reward L1 activations, complete 34-COPE L2 fixed effects, the required
nonempty condition COPEs/masks, and valid finite geometry. Canonical source exclusion
uses directory existence only. No new motion/behavior thresholds are introduced;
upstream imaging QC flags are described, not converted to exclusions. Secondary
task availability never changes primary eligibility. No raw behavior is read.

Inspect current templates and actual saved L1 design names/weights/smoothing,
plus actual L2 two-run fixed-effects input paths. Confirm specific MNI152NLin6Asym
identity from the saved L1 BOLD path and matching BOLD/COPE header grid with FEAT
registration disabled. Generic NIfTI MNI codes alone cannot identify the template
variant. Missing geometry evidence excludes that mechanical unit; conflicting
primary contrast definitions fail. Cross-task ambiguity excludes that task unit.
No source image is rewritten.

Age comes from canonical baseline participants.tsv `age`; sex defaults to `sex`
(both column names configurable). Shared Reward FlipAngle is resolved from applicable
BIDS JSON inheritance for each magnitude BOLD echo and run. Missing values remain
missing; disagreeing echoes/runs become a flagged `discrepant` category. Age,
FlipAngle, and sex never enter the voxel classifier. Baseline metadata recovery
belongs upstream, not in this repository.

## Split and folds

Select exactly 50 using StratifiedShuffleSplit, seed 20260928, trying age quintiles
× FlipAngle, quartiles, tertiles, then FlipAngle alone. Tied ages are not broken
with participant ordering to manufacture quantiles. Missing ages have their own
category. No neural data enter selection. If even FlipAngle strata fail, stop.
Five shuffled development participant folds use the same descending strata choices
when every stratum has at least five participants; otherwise explicitly recorded
participant KFold is used. Both maps always stay with their participant. The fold
file and aggregate fold demographics are hashed and preserved with the split hash.

## Exact representations

All `k` below refer to existing COPE numbers; L2 means `copek.feat/stats/cope1.nii.gz`.

| Analysis | Positive representation | Negative representation |
| --- | --- | --- |
| Primary Shared Reward L2 | ((4−3)+(6−5))/2 | 2−1 |
| Social Doors L1 run 1 | socialdoors cope4 (win−loss) | doors cope4 (win−loss) |
| Trust L2 primary | ((7−6)+(9−8))/2 | 5−4 |
| Trust friend/computer | 7−6 | 5−4 |
| Trust stranger/computer | 9−8 | 5−4 |
| Trust friend/stranger | 7−6 | 9−8 |
| Shared Reward neutral L2 | (8+9)/2 | 7 |
| Shared Reward decision L2 | (27+28)/2 | 29 |
| UGR offer modulation L2 | 14 | 13 |
| UGR social constant L2 | (5+7)/2 | (1+3)/2 |
| Shared Reward L1 reliability, each run | ((4−3)+(6−5))/2 | 2−1 |

Shared Reward reward-minus-punishment is constructed within context before social
averaging. Trust uses reciprocation-minus-defection within partner. UGR copes 13/14
sum their two endowment-specific pmods (they are not averages); the upstream EV
builder demeans offers within sociality × endowment × run. Generic social-context
controls and exploratory UGR results are reported regardless of direction, without
changing model settings. Trust decompositions are descriptive secondary results.

## Mask, geometry, normalization, and model

Use the first sorted development L2 image as the Shared Reward reference grid.
Each participant's coverage is the intersection of required L2 condition masks.
Retain voxels covered by at least 95% of development participants and intersect
with the local TemplateFlow MNI152NLin6Asym resolution-02 brain mask, resampled with
nearest-neighbor interpolation. If that exact local mask cannot be found, use
coverage alone and report the fallback; there is no network download or QC-target
mask substitution. The upstream QC mask excludes cerebellum/brainstem and is not
the whole-brain mask requested for this analysis.

The fixed mask uses all development participants without labels. Thus CV is
conditional on development-wide coverage, not a claim of fully inductive mask
estimation. Holdout masks never contribute. All primary grids must match. Verified
same-space secondary maps may be linearly resampled to the reference grid, recorded
per image under private diagnostics. Nonfinite primary voxels stop analysis;
nonfinite secondary units are reported as missing. Header-only inventory cannot
certify unread holdout intensities.

Vectorize in NumPy C-order under the saved boolean mask. Construct representations,
then subtract each representation's mean across mask voxels. No unit norm, pooled
z-scoring, feature selection, PCA, or hyperparameter search. Save image norms.
Use LinearSVC(C=1, dual=True, class_weight=None, max_iter=100000, tol=0.0001,
random_state=20260928). SOCIAL=+1, COMPUTER=−1. Nonconvergence raises an error;
there is no silently accepted unconverged model.

Train five fold models. Score every participant's primary, secondary, specificity,
and run-level maps only with the model for the fold holding that participant out.
The intercept cancels in positive-minus-negative margins. After all predictions
are saved, fit one final model on all development participants and save weights,
intercept, software versions, class orientation, repository SHAs, mask/split hashes,
and preprocessing. The final model is not used to score any RF1 participant here.

## Statistics and interpretation

Forced choice is margin > 0; exact zero is counted as incorrect and reported as a
tie. Report N, correct N, accuracy, two-sided Clopper–Pearson 95% interval, two-sided
exact binomial p versus .50, mean/median margin, and the percentile 95% interval of
10,000 participant bootstrap means (seed 20260928). No task-driven model revision
or best-algorithm selection. P values are unadjusted pilot summaries, not confirmatory
familywise claims. Overlapping CV training sets mean simple binomial/bootstrap
intervals describe participant predictions conditional on this development procedure;
they do not capture full training-set or model-selection uncertainty.

Reliability correlates out-of-fold run-1/run-2 margins: Pearson r, Spearman rho,
and ICC(3,1) = (MS_subject−MS_error)/(MS_subject+(k−1)MS_error), k=2.
This is two-way mixed **consistency**, single measurement; it intentionally ignores
systematic run offsets. Constant vectors yield undefined correlations/ICC. Also
report both run accuracies and an unlabelled scatterplot. Fold coefficients are
compared by spatial Pearson correlation under the same mask; overlapping training
samples can increase similarity. Coefficients are not activation/localization maps.

## Future external adapter interface

No external dataset or holdout scoring is implemented. A future reviewed adapter
should yield records containing `participant_id`, `positive_map`, `negative_map`,
optional `run`, and `reference_grid` (explicit MNI template identity, shape, affine,
and units). Verify identity before any resampling and reject unknown coordinate
systems. Supply an explicit permitted cohort at every data boundary.

The reusable numerical primitives are `build_mask.spatial_center(data, mask)`,
`build_mask.reconstruct(vector, mask)`, `signature_pilot.center_pair(pair)`,
and `reporting.performance(margins, analysis, seed, bootstrap_samples)`.
`signature_pilot.score_pair(pair, subject, model, guard, development_subjects,
analysis)` implements centered linear decision scoring, forced choice, and the
OOF identity check; its current guard deliberately permits only development.
An external adapter will require a separate explicit authorization/cohort guard
and a frozen model loader using `provenance/model.json` and the saved weights;
it must not bypass or loosen the RF1 holdout guard. The same centering, dot product
plus saved intercept, margin rule, and reporting functions then apply. External
scanner/sample differences belong in that future protocol, not this pilot.

Software API references: [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
for the explicit classifier settings and [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
for the exact binomial test and proportion interval. Runtime versions are recorded
with each final model; these documentation links do not replace version provenance.
