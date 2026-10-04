# Review of completed development characterization

Reviewed results commit `2343681`. This interpretation accompanies the
[generated report](REPORT.md); it does not change the analysis or authorize
validation scoring. No new fitting, source-image access, atlas download, or
anatomical exclusion was performed for this review.

The evidence supports strong condition discrimination for social context,
friend versus stranger, and outcome valence. It does not currently establish a
task-general social-reward-modulation signature, an abstract social construct,
or a reliable measure of differences between individuals.

## Completion and checks

The production run reports completion, 1,000 bootstraps per common signature,
200 permutations per specified analysis, and `holdout_scored=false`. All 101
expected characterization maps and the six main figures are present. The
recorded private membership/source-load audit passed; 270 saved-fold OOF
reconstruction checks passed (largest absolute margin error approximately
9.13e-8). Independently comparing the 645 original public result, report, and
provenance files with `f93a44b` found no changes.

All 15 permutation-null means fall between 49.28% and 50.30%. The high common
context, closeness, and valence accuracies exceed all 200 sampled null values,
giving diagnostic p=1/201=.00498. This is the resolution floor of this exercise,
not evidence that these effects have identical strength. The test does not
remove stimulus, familiarity, timing, or other condition-linked confounds.
The audit also retains its documented limitation: the fixed, label-free mask
uses coverage from all development participants, including later CV test folds.

## Begin with Figure 2: context and reward modulation separate

[Figure 2](../../../results/revised/characterization/figures/Figure2_context_reward_transfer.png)
uses the three-paradigm development cohort, N=178. All percentages below are
paired forced-choice OOF accuracy; chance is 50%.

| Model and endpoint | Shared Reward | Trust | Social/monetary Doors |
| --- | ---: | ---: | ---: |
| Common social context | 95.5% | 93.8% | 77.5% |
| Social context, entire test paradigm omitted from training | 82.0% | 95.5% | 48.9% |
| Common social reward modulation | 50.0% | 56.2% | 52.8% |

Context distinguishes human from nonsocial context averaged across outcome
valence. Social reward distinguishes human positive-minus-negative modulation
from its nonsocial counterpart. These are different scientific questions.

The common decoder's Doors accuracy includes Doors from other development
participants in training. It therefore cannot replace the decisive new-paradigm
test: Shared Reward+Trust to Doors is 48.9%, permutation p=.612. By contrast,
the two other context LOPO directions exceed their diagnostic nulls. The
asymmetry limits a universal three-paradigm claim. The common reward results
have diagnostic p=.483, .075, and .274 respectively; unsuccessful decoding is
not proof that reward modulation is absent.

## Figure 3 and full mosaics: distributed anatomy with visual contributions

[Figure 3](../../../results/revised/characterization/figures/Figure3_social_context_pattern.png)
should be read with the
[full common context mosaic](../../../results/revised/characterization/figures/three_paradigm_social_context_common_haufe_unthresholded.png)
and its
[bootstrap sign-stability mosaic](../../../results/revised/characterization/figures/three_paradigm_social_context_common_bootstrap_haufe_sign_stable_unthresholded.png).

Visual review shows positive social-associated pattern values along medial
frontal and posterior midline regions, bilateral lateral posterior regions,
and ventral posterior areas. Negative values also extend broadly, including
posterior occipital regions. This is not visibly confined to occipital or ventral
visual tissue. Those contributions nevertheless remain prominent. The
descriptive table includes lateral posterior positive peaks near
(-50.5,-61.2,29.0) and (57.5,-61.2,26.0), and a posterior negative peak near
(11.6,-72.0,11.2). These coordinates locate components; they do not identify
unique psychological processes. Large connected components span multiple
anatomical regions, so a component peak should not label its entire extent.

The task-specific Shared Reward and Trust Haufe patterns resemble one another
strongly (r=.785). Doors shares less with Shared Reward (r=.361) and moderately
with Trust (r=.539). Its task-specific mosaic has especially conspicuous
bilateral ventral posterior features. This is consistent with partial spatial
overlap plus differences in implementation; anatomy alone does not establish
why transfer is asymmetric.

There is no single answer to which task dominates the common decoder. Its raw
weights resemble Shared Reward most (r=.827), whereas its Haufe pattern and
mean-difference map resemble Doors most (r=.826 and .951). The transformations
describe different objects. Across-task amplitudes and covariance can affect
these relationships; correlations are not task-contribution estimates. The
task-specific mosaics use different color ranges, so their color intensities
must not be compared as standardized effect sizes.

Common context Haufe and mean difference correlate at r=.901, but raw weights
and Haufe correlate at only r=.178. Interpret the forward pattern as a
descriptive association with decoder output, and the weights as a multivariate
decision rule. Neither is an inferential activation map.

The social-reward Haufe patterns have very little similarity across tasks
(r=-.066 to .093). Their common bootstrap sign-stability mosaic is sparse,
unlike the extensive context pattern. Residual colored islands do not override
the weak OOF result: these are uncorrected descriptive displays, not significant
clusters.

## Figure 4: shared friend–stranger structure across outcome valence

[Figure 4](../../../results/revised/characterization/figures/Figure4_closeness_valence.png)
uses the partner-pair development cohort, N=192.

| Family | Common to Shared Reward | Common to Trust | Shared Reward to Trust | Trust to Shared Reward |
| --- | ---: | ---: | ---: | ---: |
| Friend vs stranger, positive outcomes | 76.6% | 78.1% | 75.5% | 67.7% |
| Friend vs stranger, negative outcomes | 80.2% | 83.9% | 81.8% | 75.0% |
| Friend vs stranger, positive-minus-negative modulation | 53.6% | 50.5% | 50.0% | 49.5% |
| Positive vs negative outcome | 86.5% | 95.3% | 89.6% | 77.6% |

The positive- and negative-outcome closeness Haufe patterns correlate at
r=.833; their mean differences correlate at r=.821. Both mosaics show medial
frontal, posterior midline, and lateral posterior positive friend-associated
structure. Their similarity to context is also substantial (Haufe r=.640 and
.706). This suggests shared context/partner-related pattern structure across
valence. It does not demonstrate cross-valence prediction: no decoder was
trained on positive outcomes and tested on negative outcomes in this pass.
The two maps also share participants and task implementations, so spatial
similarity is not independent replication.

Valence is spatially distinguishable from this broad structure: Haufe r=-.124
with context and -.097/-.267 with positive/negative closeness. Its mosaic shows
bilateral ventral anterior subcortical positive features and posterior visual
contributions, with a different midline distribution. These are descriptive
locations, without atlas-based labels or proof of representational independence.

The shared friend–stranger pattern can reflect closeness, familiarity, identity,
perceptual information, or relationship-linked information. The present design
does not isolate these explanations. Shared Reward is full-trial, whereas Trust
and Doors use outcome-phase maps, adding a timing difference to the stimulus
differences.

## Figures 5–6: broad condition separation, weak individual reliability

[Figure 5](../../../results/revised/characterization/figures/Figure5_oof_expression.png)
and the [margin supplement](../../../results/revised/characterization/figures/Figure5_oof_margins.png)
show that successful class ordering is widespread across participants. Context
medians are positive in all three tasks. Large margins do not receive extra
weight in forced-choice accuracy; a handful of extreme scores cannot account
for 95% correct ordering. Score magnitudes are not calibrated psychological
units and vary substantially by task.

[Figure 6](../../../results/revised/characterization/figures/Figure6_run_reliability.png)
sets a different limit. Three-paradigm context run correlations are only
r=.120 in Shared Reward and .024 in Trust. Positive-outcome closeness is the
best of these cases at .300/.223; negative closeness is .132/.085 and valence
.089/.030. A decoder can consistently order conditions within people while
poorly preserving the ranking of people across runs. These results support
condition-discrimination work, with much weaker support for trait-like scores.

The alternate QC policy preserves the qualitative performance pattern. Common
weight-map correlations across policies range from .924 to .949; context
accuracy changes by +0.4 to +3.6 percentage points, and closeness/valence by
less than 1.5 points in magnitude. This is robustness to the existing QC
sensitivity definition, not independent replication.

## Decision before validation

The strongest candidates for preregistration are Shared Reward–Trust context
and friend–stranger condition discrimination, with valence as a distinct
comparison. Keep Doors transfer as an explicit boundary and reward modulation
as an informative weak/null comparison. Do not promote all observed successful
cells to independent primary claims; specify endpoints, transfer directions,
and multiplicity before validation.

The spatial review does not justify declaring the signal purely visual or
purely abstract social information. A prospectively defined perceptual-control
analysis remains reasonable if the intended claim is abstract social coding;
it would require an independent anatomical definition or stimulus control,
not an exclusion drawn from these maps. No such analysis was run here.

The next discussion should settle the intended construct and validation
endpoints before opening any protected data. **Primary validation N=50 remains
untouched; holdout_scored=False.**
