# Empirical task-negative control

Development only: identical N=178, original five participant folds, QC and whole-brain mask for every comparison. Protected N=50 remains unscored. Reference: `monetary_doors_decision_task_negative`. All prior outputs and frozen candidates are preserved.

## Source and spatial checks

Decision COPE3 passes the modeled decision-versus-implicit-fixation audit. It is an operational estimate of task deactivation, not a resting-state or universal DMN signature. Source-code evidence alone cannot prove every historical deployment; the runtime audit additionally checks the retained runs' actual events, fitted EVs, contrasts, and geometry.

Across the five training templates: negative signed decision estimates cover 46.0–47.7% of mask voxels; DMN 89.2–90.7%; non-DMN 39.5–41.3%. Mean negative-component magnitude per voxel is 21.9–24.5 in DMN and 4.28–4.68 outside it (original COPE units, not percent signal change). Cross-fold template r=0.985–0.990; template versus phase-resolved human–computer Haufe r=-0.406–-0.296. Cross-fold training sets overlap, so their similarity is not independent reliability. Spatial correlations are descriptive, with no spatial significance claim.

The source template specifies EVs win, loss, decision and missed decision; COPE3 weights are [0, 0, 1, 0]. The task presentation draws fixation between decisions and feedback and during jittered ITIs (nominal 1.1–11.6 s). All retained runs must contain 40 decision/feedback pairs, positive interior fixation gaps, matching canonical and fitted EV timings, and an estimable decision contrast. The audit does not assume the initial/final unused acquisition time is rest. Missed-decision timing is a nuisance EV; the task code can log scheduled feedback labels even when showing a missed-response message. No new exclusion is introduced. Aggregate actual baseline timing and missed-decision counts are in `baseline_audit.tsv`.

Each training-only raw signed mean is clipped at zero *before* spatial centering and L2 normalization. Source COPEs are never precentered or sign-inverted. Negative magnitude is summarized per voxel in DMN and its complement, including zero-valued negative components. The existing Yeo-7 Default label is anatomical context only; no threshold or atlas is selected from performance.

The Haufe comparator is a descriptive fixed-C human–computer fit pooling positive/negative outcome maps from Shared Reward and Trust on each same fold-training set. It is not a new frozen candidate. Full-N display maps are separately labelled `DEV_display`; none enters CV. Signed response COPE, centered unit direction, and Haufe covariance/score pattern have distinct units. Positive voxels in the centered direction need not represent positive task-versus-baseline responses.

## Matched transfers

| Transfer | Original | Task-negative expression | Projection removed |
|---|---|---|---|
| SR → Trust: human–computer (primary) | 82.4% [79.1, 85.7] | 82.0% [77.8, 86.2] | 80.9% [77.4, 84.3] |
| Trust → SR: human–computer (primary) | 70.6% [66.4, 74.7] | 61.0% [55.9, 66.0] | 69.8% [65.7, 73.9] |
| SR → Trust: friend–stranger (secondary) | 73.3% [69.4, 77.1] | 77.2% [72.8, 81.7] | 72.1% [68.1, 75.8] |
| Trust → SR: friend–stranger (secondary) | 65.7% [61.5, 69.9] | 61.8% [57.0, 66.6] | 65.2% [61.2, 69.1] |
| SR → Doors: social–monetary (secondary) | 73.0% [68.3, 77.5] | 68.0% [62.4, 73.9] | 73.5% [68.8, 77.9] |
| Trust → Doors: social–monetary (secondary) | 70.4% [65.2, 75.6] | 68.0% [62.4, 73.9] | 71.8% [66.7, 76.7] |

| Primary comparison versus original | Change (pp; 95% CI) | Above-chance permutation p | Two-sided maxT p | Null mean ± SD |
|---|---|---|---|---|
| SR → Trust: human–computer / Original outcome SVM | reference | 0.0020 | 0.0020 | 0.500 ± 0.019 |
| SR → Trust: human–computer / Task-negative one-feature | -0.4 [-5.8, +4.9] | 0.0020 | 0.0020 | 0.501 ± 0.023 |
| SR → Trust: human–computer / Projection-removed SVM | -1.5 [-2.5, -0.7] | 0.0020 | 0.0020 | 0.500 ± 0.020 |
| Trust → SR: human–computer / Original outcome SVM | reference | 0.0020 | 0.0020 | 0.501 ± 0.020 |
| Trust → SR: human–computer / Task-negative one-feature | -9.7 [-14.6, -4.8] | 0.0020 | 0.0020 | 0.500 ± 0.022 |
| Trust → SR: human–computer / Projection-removed SVM | -0.8 [-2.7, +0.8] | 0.0020 | 0.0020 | 0.501 ± 0.020 |

Accuracy is paired ordering, averaging correctness across four train/test-valence cells within each person. Original outcome-only margins must reproduce the completed specificity run (tolerance 1e-5 absolute / 2e-5 relative, identical correctness). Source features, centering, C=1, labels, orientations and voxel-model dual solver are unchanged. The independent spatial template uses no social labels; its one-dimensional C=1 readout uses training labels and the fixed primal solver, as in the prior one-feature baseline. Residualization removes exactly one centered unit direction from training and test maps before refitting the original SVM.

## Raw expression direction

| Outcome domain | First − second mean [95% CI] |
|---|---|
| sharedreward_positive | -1.07e+03 [-1.64e+03, -486] |
| sharedreward_negative | -1.01e+03 [-1.73e+03, -312] |
| trust_positive | -1.36e+03 [-1.59e+03, -1.13e+03] |
| trust_negative | -1.32e+03 [-1.56e+03, -1.08e+03] |
| socialdoors_positive | -1.86e+03 [-2.54e+03, -1.21e+03] |
| socialdoors_negative | -2.02e+03 [-2.69e+03, -1.39e+03] |
| sr_friend_stranger_positive | -1.37e+03 [-2.24e+03, -492] |
| sr_friend_stranger_negative | -1.2e+03 [-1.99e+03, -413] |
| trust_friend_stranger_positive | -1.02e+03 [-1.28e+03, -776] |
| trust_friend_stranger_negative | -1.64e+03 [-1.9e+03, -1.38e+03] |

First/second means human/computer, friend/stranger, or social/monetary Doors, respectively. Positive differences mean greater alignment with the centered task-negative direction in the first condition; they are signed dot products, not absolute deactivation relative to fixation in that social condition. Positive/negative valences remain separate; both condition-score means and intervals are in `raw_signed_expression.tsv`.

## Inference and interpretation

500 synchronized participant-label flips per variant, refitting each fold; the same flip applies to every task/valence and test label for that person. Existing original nulls are reused only after benchmark reconstruction and numerical reproduction of the first two null replicates. Templates remain fixed under permutations because construction is label-independent. MaxT uses all six primary endpoint × variant combinations as a complete-null familywise diagnostic, not a test of model differences. Null-centering flags (>0.03 absolute mean deviation): 0/6. Full null quantiles, Monte Carlo tail intervals, unique values and centering diagnostics are in `transfers.tsv`; minimum p=0.0020. No secondary permutation matrix is launched.

Performance and matched-difference intervals use 10,000 participant-bootstrap resamples, keeping each person’s four valence cells together and identical resamples across variants. These fixed-prediction intervals omit uncertainty from refitting models and templates, and are descriptive. The original label-free development coverage mask includes CV test participants; it remains unchanged and is not training-fold-specific.

SR → Trust: human–computer: expression alone achieves 82.0%; residual decoding achieves 80.9% (change -1.5 pp, CI -2.5 to -0.7).
Trust → SR: human–computer: expression alone achieves 61.0%; residual decoding achieves 69.8% (change -0.8 pp, CI -2.7 to +0.8).

Interpret these estimates and intervals against three possibilities: **A**, strong expression prediction plus a substantial residual drop supports a generic deactivation explanation; **B**, weak expression prediction with retained residual decoding supports information beyond this direction; **C**, expression prediction alongside retained residual decoding supports partial overlap. No new cutoff is imposed to force an A/B/C label, and a nonsignificant difference does not establish equivalence. A one-direction projection cannot remove all generic task signals. DMN anatomy and task-negative function are not synonymous; neither survival nor anatomical location proves abstract social coding.

Social Doors transfers are secondary: training-only monetary Doors decisions supply the template while monetary Doors outcomes enter the target comparison. People remain separated and phases are modeled separately, but these endpoints share task context and are not independent template validation.

The observed pattern supports **partial overlap**: the independent task-negative direction strongly predicts SR→Trust, while both transfers retain substantial decoding after its removal. This does not show equivalence of the models or exclude other generic task signals.

## Decisions for preregistration — stop development here

1. Agree whether the claim is human–computer social context, friend–stranger closeness, generic deactivation, or overlapping information; state the limits of the task manipulations.
2. Choose among the already documented candidate strategies and fix the final source phase, contrast, training sample and scoring method. This control does not replace the frozen candidates or select a residualized/atlas-restricted model.
3. Fix the validation datasets, contrast mapping, QC/coverage rules, primary endpoint/direction, multiplicity correction and success criterion before accessing validation outcomes.
4. Preregister the protected N=50 execution plan and external validation. No UGR/betrayal, additional template, atlas, threshold or parameter search follows this report.

[Comparison figure](../../../results/revised/task_negative_control/figures/task_negative_comparison.png). All statistics are development results.

Reporting recovery: all 1,000 new permutation jobs completed on Linux2 before a Nilearn/Matplotlib import incompatibility stopped plotting. This report and figure were rendered from the hash-verified committed aggregates and development group maps; no participant images, model fits, bootstrap resamples or new permutations were used. The original failed calculation status is retained for provenance. `render_status.json` records completion of this reporting-only recovery.
