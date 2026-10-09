# Final development specificity

N=178 for every model and comparison; original five folds, QC and 95% development-coverage mask retained. Validation remains unscored. No UGR/betrayal models executed.

## Matched central results

| Transfer | Previous SR full-trial | Outcomes | DMN only | DMN excluded | Generic template |
|---|---|---|---|---|---|
| SR → Trust | 85.8% [82.7, 88.8] | 82.4% [79.1, 85.7] | 73.5% [69.7, 77.0] | 80.5% [76.8, 84.0] | 50.8% [45.2, 56.5] |
| SR → Doors | 45.6% [39.6, 51.8] | 73.0% [68.3, 77.7] | 55.3% [50.1, 60.4] | 73.9% [69.1, 78.4] | 46.6% [40.2, 53.1] |
| Trust → SR | 79.4% [75.4, 83.3] | 70.6% [66.4, 74.7] | 65.4% [61.5, 69.4] | 69.2% [65.0, 73.5] | 60.4% [55.3, 65.4] |
| Trust → Doors | 70.4% [65.0, 75.6] | 70.4% [65.0, 75.6] | 61.1% [55.6, 66.3] | 69.5% [64.3, 74.6] | 55.1% [48.6, 61.2] |
| Doors → SR | 62.5% [58.0, 67.0] | 66.4% [62.1, 70.6] | 59.7% [55.2, 64.2] | 66.6% [62.2, 70.8] | 62.6% [57.6, 67.7] |
| Doors → Trust | 77.2% [73.0, 81.2] | 77.2% [73.2, 81.3] | 63.6% [58.7, 68.5] | 74.9% [70.4, 79.2] | 74.7% [69.7, 79.5] |
| Doors → Doors | 93.1% [89.7, 96.1] | 93.1% [89.7, 96.1] | 86.4% [82.6, 89.9] | 93.3% [89.9, 96.2] | 57.3% [50.6, 64.0] |
| SR (F−S) → Trust (F−S) | 79.9% [76.5, 83.1] | 73.3% [69.4, 77.2] | 69.4% [65.4, 73.2] | 70.9% [66.9, 74.9] | 44.1% [38.8, 49.7] |
| Trust (F−S) → SR (F−S) | 70.1% [65.7, 74.3] | 65.7% [61.5, 69.8] | 60.5% [56.5, 64.5] | 64.9% [61.0, 68.7] | 64.3% [59.3, 69.4] |

| Outcome transfer | Change vs legacy (pp, 95% CI) | Permutation p (above chance) | Corrected two-sided p | Null mean ± SD |
|---|---|---|---|---|
| SR → Trust | -3.4 [-6.6, -0.1] | 0.0020 | 0.0020 | 0.500 ± 0.019 |
| SR → Doors | +27.4 [+20.6, +33.8] | 0.0020 | 0.0020 | 0.502 ± 0.023 |
| Trust → SR | -8.7 [-12.2, -5.2] | 0.0020 | 0.0020 | 0.501 ± 0.020 |
| Trust → Doors | +0.0 [+0.0, +0.0] | 0.0020 | 0.0020 | 0.501 ± 0.025 |
| Doors → SR | +3.9 [-0.3, +8.1] | 0.0020 | 0.0020 | 0.502 ± 0.023 |
| Doors → Trust | +0.0 [+0.0, +0.0] | 0.0020 | 0.0020 | 0.500 ± 0.024 |
| Doors → Doors | +0.0 [+0.0, +0.0] | 0.0020 | 0.0020 | 0.501 ± 0.038 |
| SR (F−S) → Trust (F−S) | -6.6 [-9.8, -3.4] | 0.0020 | 0.0020 | 0.500 ± 0.020 |
| Trust (F−S) → SR (F−S) | -4.4 [-8.6, -0.3] | 0.0020 | 0.0020 | 0.498 ± 0.019 |

Accuracy is paired ordering: score(human)>score(computer), or score(friend)>score(stranger). Each transfer averages correctness across the four positive/negative training/testing cells within participant; all 52 individual cells per model are in `valence_matrix.tsv`. Doors→Doors retains same- and cross-valence positive controls. Trust and Doors inputs are unchanged; only SR switches from full-trial to outcome-only. Previous social-context OOF predictions must reproduce before any interpretation. F−S denotes friend–stranger transfers on the same matched N=178; its legacy comparison is a new matched-cohort refit, not the previous N=192 estimate.

## Inference and scope

500 synchronized participant label-flip permutations per variant refit every fold, including generic-template score orientation. Central-transfer diagnostics report above-chance and two-sided p values, max-|accuracy−0.5| correction across all 45 prespecified endpoint/variant combinations, null means/SD/quantiles, unique values, and Monte Carlo tail intervals. Max-statistic correction is a complete-null familywise diagnostic. Minimum p resolution is 0.0020. Null-centering flags: 0/45 (descriptive absolute mean deviation >0.03; no rerun/selection rule).

Paired participant-bootstrap intervals for outcome−legacy and each baseline−outcome are in `paired_comparisons.tsv`. These are descriptive intervals conditional on the fitted OOF predictions; they do not capture full retraining uncertainty. Cellwise intervals are exact binomial descriptive OOF intervals. No bootstrap unit is a map or fold. Permutation evidence for above-chance discrimination is not a test of differences between model variants.

## Fixed references and controls

Yeo 2011 seven-network liberal cortical atlas, label 7 (Default), nearest-neighbor resampled once into the frozen MNI152NLin6Asym grid. Analysis-mask intersection: 10,922/84,045 voxels. This resting-state cortical DMN reference does not exhaust task-negative regions; DMN-excluded retains all other original-mask voxels, including unlabelled/subcortical voxels. No dilation, accuracy-selected threshold, mask search or C tuning.

Generic task-response template: source-task training participants only, equal mean of six SR/Trust partner×valence COPEs (four social/monetary×valence Doors COPEs), spatially centered and unit norm. A fixed C=1 one-dimensional LinearSVC uses the primal solver (dual=False; same L2 squared-hinge objective, tolerance and iteration limit) and learns orientation from training projections; held-out participants contribute neither template nor orientation. This is an internal, label-blind direction control, not an external population task-negative template.

All models retain independent per-map spatial centering (reapplied within restricted masks), without population feature scaling. The retained whole mask was defined using all development subjects, including subsequent CV test subjects; it is label-free but not training-fold-specific. Restricted masks therefore test different feature spaces, not equivalent model capacities.

Frozen-candidate overlap characterizes the unchanged N=192 development fits, not held-out predictive performance. It uses original candidate masks and unchanged weights/forward patterns/mean differences: absolute and squared spatial-mass fractions, enrichment relative to DMN voxel fraction, signed spatial correlation with the binary reference. These threshold-free descriptive measures are not spatial significance tests. DMN overlap or survival outside DMN cannot establish an abstract social representation.

## Remaining preregistration decisions

1. Confirm the primary construct and contrast (human–computer versus friend–stranger), and whether the already-frozen full-trial candidates remain primary or an outcome-only candidate needs a separately documented freeze. This package does not replace either candidate.
2. Specify the external validation dataset, task/contrast mapping, QC and coverage rules, and primary endpoint/direction before accessing validation outcomes.
3. Fix the confirmatory hypothesis family, multiplicity policy and success criterion; classify Doors and DMN checks as specificity evidence without choosing masks from these accuracies.
4. Approve the protected N=50 scoring plan and execution timing. No protected images or scores were accessed here.

[Comparison figure](../../../results/revised/specificity/figures/specificity_comparison.png). [Independent atlas documentation](https://freesurfer.net/fswiki/CorticalParcellation_Yeo2011).

Stop here: no additional exploratory analysis is launched.
