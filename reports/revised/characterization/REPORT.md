# Frozen v4 development-result characterization

This report characterizes the existing primary `tsnr_coverage_fd` development results. No original assignments, folds, masks, contrasts, QC thresholds, or classifier settings were changed. The four-metric policy is a sensitivity analysis. No validation data were loaded or scored.

## Audit and sample

See [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md). Private membership and frozen hashes passed; selected saved fold maps reproduced OOF margins and classifications before scientific characterization. Primary development N=192 for the partner pair and N=178 for three paradigms. QC-qualified primary validation N=50 remains untouched.

## What the decoders track spatially

Raw LinearSVC coefficients are discriminative weights, not activation or independent regional effects. Haufe patterns describe covariance between the exact centered training maps and the fitted decoder output, divided by output variance. They are descriptive forward patterns, not inferential or causal maps. Mean-difference maps describe group-average class differences. The correlations below show whether these three descriptions align; they do not establish a psychological mechanism.

| cohort | family | model | n_participants | n_training_maps | weight_haufe_r | weight_mean_r | haufe_mean_r |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three_paradigm | social_context | common | 178 | 1068 | 0.1779 | 0.2091 | 0.9007 |
| three_paradigm | social_reward | common | 178 | 1068 | 0.3562 | 0.422 | 0.8971 |
| partner_pair | closeness_positive | common | 192 | 768 | 0.2688 | 0.3681 | 0.9457 |
| partner_pair | closeness_negative | common | 192 | 768 | 0.2175 | 0.2853 | 0.9718 |
| partner_pair | valence | common | 192 | 768 | 0.3409 | 0.4136 | 0.9532 |

## Common versus task-specific patterns

Task-specific patterns are compared on their fixed common mask. Correlations with the pooled pattern indicate resemblance, not proof that one task dominates the decoder; correlated predictors and task scaling can affect raw weights.

| family | kind | model_a | model_b | spatial_r |
| --- | --- | --- | --- | --- |
| social_context | weight | sharedreward | trust | 0.1316 |
| social_context | weight | sharedreward | socialdoors | -0.04597 |
| social_context | weight | socialdoors | trust | 0.04194 |
| social_context | haufe | sharedreward | trust | 0.7852 |
| social_context | haufe | sharedreward | socialdoors | 0.3608 |
| social_context | haufe | socialdoors | trust | 0.5394 |
| social_context | mean_difference | sharedreward | trust | 0.7668 |
| social_context | mean_difference | sharedreward | socialdoors | 0.3402 |
| social_context | mean_difference | socialdoors | trust | 0.5237 |
| social_reward | weight | sharedreward | trust | 0.03277 |
| social_reward | weight | sharedreward | socialdoors | -0.0107 |
| social_reward | weight | socialdoors | trust | 0.02172 |
| social_reward | haufe | sharedreward | trust | 0.03461 |
| social_reward | haufe | sharedreward | socialdoors | -0.06586 |
| social_reward | haufe | socialdoors | trust | 0.09322 |
| social_reward | mean_difference | sharedreward | trust | 0.03638 |
| social_reward | mean_difference | sharedreward | socialdoors | -0.05863 |
| social_reward | mean_difference | socialdoors | trust | 0.09339 |

Common-model comparisons:

| family | kind | model_a | model_b | spatial_r |
| --- | --- | --- | --- | --- |
| social_context | weight | sharedreward | common | 0.8267 |
| social_context | weight | trust | common | 0.3588 |
| social_context | weight | socialdoors | common | 0.185 |
| social_context | haufe | sharedreward | common | 0.6627 |
| social_context | haufe | trust | common | 0.7694 |
| social_context | haufe | socialdoors | common | 0.8264 |
| social_context | mean_difference | sharedreward | common | 0.5704 |
| social_context | mean_difference | trust | common | 0.7576 |
| social_context | mean_difference | socialdoors | common | 0.9507 |
| social_reward | weight | sharedreward | common | 0.8229 |
| social_reward | weight | trust | common | 0.3536 |
| social_reward | weight | socialdoors | common | 0.2334 |
| social_reward | haufe | sharedreward | common | 0.2065 |
| social_reward | haufe | trust | common | 0.4865 |
| social_reward | haufe | socialdoors | common | 0.7603 |
| social_reward | mean_difference | sharedreward | common | 0.168 |
| social_reward | mean_difference | trust | common | 0.5681 |
| social_reward | mean_difference | socialdoors | common | 0.8484 |

For social_context, the common Haufe pattern is most similar to socialdoors (r=0.826) and least similar to sharedreward (r=0.663). This is a descriptive spatial comparison, not a task-contribution estimate.

For social_reward, the common Haufe pattern is most similar to socialdoors (r=0.760) and least similar to sharedreward (r=0.206). This is a descriptive spatial comparison, not a task-contribution estimate.

## Context, closeness, reward modulation, and valence

The performance dissociation must retain its original meaning: social context averages across outcome valence; social reward compares human versus nonsocial positive-minus-negative modulation; closeness is friend versus stranger; valence is positive versus negative outcome. Strong context and friend–stranger decoding cannot be relabeled as a task-general social-reward signature. Failed reward decoding is not proof of no reward-related information.

Common-model cross-family similarities use intersecting valid masks (resampling only for comparison if needed):

| family_a | family_b | kind | spatial_r | intersection_voxels | resampled_for_comparison |
| --- | --- | --- | --- | --- | --- |
| social_context | social_reward | weight | -0.0133 | 84045 | False |
| social_context | social_reward | haufe | -0.1076 | 84045 | False |
| social_context | social_reward | mean_difference | -0.0769 | 84045 | False |
| social_context | closeness_positive | weight | 0.1397 | 84032 | False |
| social_context | closeness_positive | haufe | 0.6402 | 84032 | False |
| social_context | closeness_positive | mean_difference | 0.6136 | 84032 | False |
| social_context | closeness_negative | weight | 0.1561 | 84032 | False |
| social_context | closeness_negative | haufe | 0.706 | 84032 | False |
| social_context | closeness_negative | mean_difference | 0.6885 | 84032 | False |
| social_context | valence | weight | -0.02964 | 84032 | False |
| social_context | valence | haufe | -0.1236 | 84032 | False |
| social_context | valence | mean_difference | -0.1528 | 84032 | False |
| social_reward | closeness_positive | weight | -0.01658 | 84032 | False |
| social_reward | closeness_positive | haufe | -0.07743 | 84032 | False |
| social_reward | closeness_positive | mean_difference | -0.1869 | 84032 | False |
| social_reward | closeness_negative | weight | -0.0003382 | 84032 | False |
| social_reward | closeness_negative | haufe | -0.1235 | 84032 | False |
| social_reward | closeness_negative | mean_difference | -0.2257 | 84032 | False |
| social_reward | valence | weight | 0.2197 | 84032 | False |
| social_reward | valence | haufe | 0.1873 | 84032 | False |
| social_reward | valence | mean_difference | 0.1534 | 84032 | False |
| closeness_positive | closeness_negative | weight | 0.1405 | 84133 | False |
| closeness_positive | closeness_negative | haufe | 0.833 | 84133 | False |
| closeness_positive | closeness_negative | mean_difference | 0.8209 | 84133 | False |
| closeness_positive | valence | weight | -0.008155 | 84133 | False |
| closeness_positive | valence | haufe | -0.09735 | 84133 | False |
| closeness_positive | valence | mean_difference | -0.1354 | 84133 | False |
| closeness_negative | valence | weight | -0.002591 | 84133 | False |
| closeness_negative | valence | haufe | -0.2674 | 84133 | False |
| closeness_negative | valence | mean_difference | -0.2602 | 84133 | False |

Positive- versus negative-outcome closeness weight: r=0.140.

Positive- versus negative-outcome closeness haufe: r=0.833.

Positive- versus negative-outcome closeness mean_difference: r=0.821.

## Descriptive spatial locations and alternatives

Bootstrap sign stability uses 1000 participant resamples and P(positive) or P(negative) ≥0.975. Components have at least 20 face-connected voxels. Peaks and means below refer to the bootstrap-mean Haufe pattern. No FWE/FDR correction or cluster p value is implied. Display thresholds (top 5%/1%) are also descriptive. No atlas was downloaded; coordinates alone do not establish a network assignment.

| cohort | family | sign | voxel_count | peak_absolute_haufe | peak_signed_haufe | x | y | z | mean_haufe | connectivity | inferential_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| partner_pair | closeness_negative | 1 | 9926 | 28.71 | 28.71 | 0.8 | 9 | -12.6 | 6.513 | 6 | not_applicable |
| three_paradigm | social_context | 1 | 13052 | 24.18 | 24.18 | -1.9 | -45 | 5.22 | 6.401 | 6 | not_applicable |
| partner_pair | closeness_positive | 1 | 1926 | 23.92 | 23.92 | 0.8 | -50.4 | 23.04 | 7.313 | 6 | not_applicable |
| partner_pair | closeness_negative | 1 | 3234 | 18.59 | 18.59 | -31.6 | 17.1 | -15.57 | 5.993 | 6 | not_applicable |
| partner_pair | closeness_positive | 1 | 4047 | 18.43 | 18.43 | -1.9 | 60.3 | 17.1 | 5.227 | 6 | not_applicable |
| partner_pair | closeness_positive | 1 | 174 | 18.29 | 18.29 | 0.8 | 6.3 | -12.6 | 5.685 | 6 | not_applicable |
| three_paradigm | social_context | 1 | 693 | 17.57 | 17.57 | 57.5 | -61.2 | 26.01 | 7.306 | 6 | not_applicable |
| three_paradigm | social_context | -1 | 18898 | 16.94 | -16.94 | 11.6 | -72 | 11.16 | -4.394 | 6 | not_applicable |
| partner_pair | closeness_negative | -1 | 21436 | 16.34 | -16.34 | -53.2 | -34.2 | 58.68 | -3.872 | 6 | not_applicable |
| three_paradigm | social_reward | 1 | 36 | 16.28 | 16.28 | 0.8 | -50.4 | -48.24 | 9.361 | 6 | not_applicable |
| three_paradigm | social_reward | 1 | 63 | 15.86 | 15.86 | -10 | -39.6 | -51.21 | 7.577 | 6 | not_applicable |
| partner_pair | closeness_negative | 1 | 2327 | 15.14 | 15.14 | 6.2 | -50.4 | -45.27 | 4.769 | 6 | not_applicable |
| partner_pair | closeness_negative | -1 | 354 | 14.9 | -14.9 | -10 | -36.9 | -54.18 | -3.558 | 6 | not_applicable |
| partner_pair | closeness_positive | -1 | 111 | 14.8 | -14.8 | 0.8 | -20.7 | -48.24 | -5.93 | 6 | not_applicable |
| partner_pair | closeness_positive | -1 | 12045 | 14.68 | -14.68 | 62.9 | 3.6 | 2.25 | -3.639 | 6 | not_applicable |
| three_paradigm | social_context | 1 | 810 | 13.85 | 13.85 | -50.5 | -61.2 | 28.98 | 6.147 | 6 | not_applicable |
| partner_pair | closeness_positive | 1 | 798 | 13.16 | 13.16 | -47.8 | -61.2 | 34.92 | 5.901 | 6 | not_applicable |
| three_paradigm | social_reward | -1 | 38 | 12.74 | -12.74 | -47.8 | 19.8 | -6.66 | -7.529 | 6 | not_applicable |
| partner_pair | closeness_positive | 1 | 249 | 12.68 | 12.68 | 6.2 | -50.4 | -42.3 | 5.706 | 6 | not_applicable |
| three_paradigm | social_reward | -1 | 49 | 12.17 | -12.17 | 52.1 | 19.8 | -6.66 | -7.062 | 6 | not_applicable |
| three_paradigm | social_reward | -1 | 37 | 11.36 | -11.36 | 8.9 | -90.9 | -18.54 | -8.376 | 6 | not_applicable |
| partner_pair | valence | 1 | 284 | 10.95 | 10.95 | -12.7 | 6.3 | -12.6 | 4.416 | 6 | not_applicable |
| partner_pair | closeness_positive | -1 | 117 | 10.67 | -10.67 | 11.6 | -1.8 | -21.51 | -5.047 | 6 | not_applicable |
| partner_pair | valence | 1 | 456 | 10.52 | 10.52 | 11.6 | 11.7 | -6.66 | 4.201 | 6 | not_applicable |
| partner_pair | closeness_negative | -1 | 45 | 10.3 | -10.3 | -12.7 | -1.8 | -24.48 | -6.089 | 6 | not_applicable |
| partner_pair | closeness_positive | 1 | 1440 | 10.29 | 10.29 | 49.4 | -55.8 | 26.01 | 3.943 | 6 | not_applicable |
| three_paradigm | social_context | -1 | 130 | 10.17 | -10.17 | -4.6 | -15.3 | -36.36 | -4.561 | 6 | not_applicable |
| partner_pair | closeness_positive | 1 | 411 | 10 | 10 | -31.6 | 14.4 | -18.54 | 4.403 | 6 | not_applicable |
| three_paradigm | social_reward | -1 | 48 | 9.957 | -9.957 | 33.2 | 17.1 | -18.54 | -6.389 | 6 | not_applicable |
| three_paradigm | social_reward | -1 | 29 | 9.877 | -9.877 | 0.8 | 38.7 | -0.72 | -6.234 | 6 | not_applicable |
| three_paradigm | social_context | -1 | 372 | 9.855 | -9.855 | -1.9 | -34.2 | -6.66 | -4.735 | 6 | not_applicable |
| partner_pair | closeness_positive | -1 | 481 | 9.802 | -9.802 | -39.7 | 52.2 | 23.04 | -4.52 | 6 | not_applicable |
| three_paradigm | social_reward | 1 | 95 | 9.766 | 9.766 | -28.9 | -47.7 | -9.63 | 5.125 | 6 | not_applicable |
| partner_pair | valence | -1 | 217 | 9.577 | -9.577 | 3.5 | -34.2 | 2.25 | -3.91 | 6 | not_applicable |
| three_paradigm | social_reward | -1 | 29 | 9.526 | -9.526 | -31.6 | -63.9 | 64.62 | -6.071 | 6 | not_applicable |
| three_paradigm | social_reward | -1 | 35 | 9.504 | -9.504 | -18.1 | -90.9 | -24.48 | -5.943 | 6 | not_applicable |
| partner_pair | closeness_positive | -1 | 65 | 9.282 | -9.282 | -12.7 | -1.8 | -21.51 | -5.307 | 6 | not_applicable |
| partner_pair | closeness_positive | 1 | 722 | 9.142 | 9.142 | -58.6 | -12.6 | -12.6 | 3.692 | 6 | not_applicable |
| three_paradigm | social_reward | 1 | 54 | 9.112 | 9.112 | 27.8 | -42.3 | -9.63 | 5.253 | 6 | not_applicable |
| three_paradigm | social_reward | 1 | 38 | 8.896 | 8.896 | -1.9 | 65.7 | -6.66 | 5.476 | 6 | not_applicable |

Inspect Figure 3 and the full slice mosaics alongside these coordinates for occipital/ventral versus broader distributions. An automated coordinate table cannot resolve sensory versus abstract social coding, so anatomical/network attribution remains a visual-review item. Friend–stranger patterns can reflect closeness, familiarity, person identity, perceptual information, and relationship-linked representations. Shared Reward is full-trial; Trust and Doors are outcome-phase. Shared visual content and different event timing remain alternative explanations. If visual review indicates a predominantly perceptual pattern, an independently defined, prospectively justified visual-control sensitivity analysis is a next-step option; none was implemented here.

## Participant-label permutation sanity checks

200 permutations refit the relevant five-fold classifiers with one random label flip per participant, shared across that participant’s tasks. Both training and test labels are permuted. Folds, features, masks and parameters remain fixed. Empirical upper-tail p=(1+null≥observed)/(1+B); these diagnostic values do not replace the original inferential framework or provide project-wide multiplicity control.

| cohort | family | scope | test_task | n | permutations | observed_accuracy | null_mean | null_sd | null_mean_mc_se | null_q025 | null_q975 | empirical_p | chance_centering_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three_paradigm | social_context | common | sharedreward | 178 | 200 | 0.9551 | 0.5024 | 0.04601 | 0.003254 | 0.4101 | 0.6011 | 0.004975 | False |
| three_paradigm | social_context | common | trust | 178 | 200 | 0.9382 | 0.4975 | 0.03927 | 0.002777 | 0.4268 | 0.5677 | 0.004975 | False |
| three_paradigm | social_context | common | socialdoors | 178 | 200 | 0.7753 | 0.4992 | 0.03932 | 0.002781 | 0.4324 | 0.5788 | 0.004975 | False |
| three_paradigm | social_context | lopo | sharedreward | 178 | 200 | 0.8202 | 0.4994 | 0.03723 | 0.002632 | 0.4324 | 0.5732 | 0.004975 | False |
| three_paradigm | social_context | lopo | trust | 178 | 200 | 0.9551 | 0.4976 | 0.03566 | 0.002521 | 0.4382 | 0.5619 | 0.004975 | False |
| three_paradigm | social_context | lopo | socialdoors | 178 | 200 | 0.4888 | 0.4974 | 0.03703 | 0.002619 | 0.4326 | 0.573 | 0.6119 | False |
| three_paradigm | social_reward | common | sharedreward | 178 | 200 | 0.5 | 0.4928 | 0.04435 | 0.003136 | 0.4044 | 0.5844 | 0.4826 | False |
| three_paradigm | social_reward | common | trust | 178 | 200 | 0.5618 | 0.503 | 0.04121 | 0.002914 | 0.4212 | 0.5732 | 0.07463 | False |
| three_paradigm | social_reward | common | socialdoors | 178 | 200 | 0.5281 | 0.499 | 0.03958 | 0.002799 | 0.4212 | 0.573 | 0.2736 | False |
| partner_pair | closeness_positive | common | sharedreward | 192 | 200 | 0.7656 | 0.502 | 0.04488 | 0.003173 | 0.4167 | 0.5939 | 0.004975 | False |
| partner_pair | closeness_positive | common | trust | 192 | 200 | 0.7812 | 0.5029 | 0.0398 | 0.002814 | 0.4271 | 0.5783 | 0.004975 | False |
| partner_pair | closeness_negative | common | sharedreward | 192 | 200 | 0.8021 | 0.5022 | 0.04464 | 0.003156 | 0.406 | 0.5784 | 0.004975 | False |
| partner_pair | closeness_negative | common | trust | 192 | 200 | 0.8385 | 0.4986 | 0.03962 | 0.002801 | 0.4165 | 0.5729 | 0.004975 | False |
| partner_pair | valence | common | sharedreward | 192 | 200 | 0.8646 | 0.5008 | 0.04178 | 0.002954 | 0.4271 | 0.5833 | 0.004975 | False |
| partner_pair | valence | common | trust | 192 | 200 | 0.9531 | 0.501 | 0.03618 | 0.002558 | 0.4375 | 0.573 | 0.004975 | False |

All empirical null means are within .05 of .50 under the prespecified descriptive centering check. This checks the label-scrambling implementation, not every possible confound.

## OOF score distributions

Figure 5 uses only verified out-of-fold scores. Final in-sample DEV scores are used only in the descriptive Haufe transformation, never as accuracy evidence. Forced-choice accuracy counts the sign of each participant’s margin equally: extreme margins cannot by magnitude alone create high accuracy. Means can be influenced by extremes; medians, individual margins, and quantiles are provided below.

| cohort | family | task | n | positive_fraction | mean | median | mean_ci_low | mean_ci_high | q05 | q95 | minimum | maximum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three_paradigm | social_context | sharedreward | 178 | 0.9551 | 0.9886 | 0.9747 | 0.8939 | 1.083 | 0.04474 | 2.098 | -0.5373 | 2.978 |
| three_paradigm | social_context | socialdoors | 178 | 0.7753 | 3.482 | 3.17 | 2.693 | 4.251 | -4.579 | 12.64 | -16.95 | 18.08 |
| three_paradigm | social_context | trust | 178 | 0.9382 | 2.177 | 1.98 | 1.96 | 2.395 | -0.1246 | 4.573 | -1.699 | 6.632 |
| three_paradigm | social_reward | sharedreward | 178 | 0.5 | -0.01471 | -0.0001424 | -0.07557 | 0.04618 | -0.6163 | 0.6748 | -1.571 | 1.394 |
| three_paradigm | social_reward | socialdoors | 178 | 0.5281 | 0.07712 | 0.1288 | -0.1953 | 0.3504 | -3.327 | 3.331 | -5.318 | 5.966 |
| three_paradigm | social_reward | trust | 178 | 0.5618 | 0.09666 | 0.1501 | -0.04681 | 0.2398 | -1.655 | 1.563 | -2.776 | 3.693 |
| partner_pair | closeness_positive | sharedreward | 192 | 0.7656 | 0.4673 | 0.4227 | 0.3705 | 0.5651 | -0.5947 | 1.481 | -1.371 | 3.736 |
| partner_pair | closeness_positive | trust | 192 | 0.7812 | 1.112 | 1.014 | 0.9044 | 1.325 | -1.234 | 3.219 | -3.194 | 8.257 |
| partner_pair | closeness_negative | sharedreward | 192 | 0.8021 | 0.5073 | 0.4715 | 0.4124 | 0.6014 | -0.5878 | 1.672 | -1.389 | 2.445 |
| partner_pair | closeness_negative | trust | 192 | 0.8385 | 1.501 | 1.478 | 1.283 | 1.72 | -1.015 | 3.943 | -2.547 | 6.64 |
| partner_pair | valence | sharedreward | 192 | 0.8646 | 0.5571 | 0.5735 | 0.4866 | 0.6271 | -0.2952 | 1.389 | -0.6947 | 2.248 |
| partner_pair | valence | trust | 192 | 0.9531 | 2.024 | 1.904 | 1.829 | 2.22 | 0.01734 | 4.573 | -1.472 | 5.572 |

## QC robustness

| cohort | family | task | primary_n | sensitivity_n | primary_accuracy | sensitivity_accuracy | accuracy_difference_sensitivity_minus_primary | raw_weight_r | intersection_voxels | resampled_for_comparison |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three_paradigm | social_context | sharedreward | 178 | 148 | 0.9551 | 0.9595 | 0.004403 | 0.9246 | 84035 | False |
| three_paradigm | social_context | trust | 178 | 148 | 0.9382 | 0.9459 | 0.007744 | 0.9246 | 84035 | False |
| three_paradigm | social_context | socialdoors | 178 | 148 | 0.7753 | 0.8108 | 0.03553 | 0.9246 | 84035 | False |
| three_paradigm | social_reward | sharedreward | 178 | 148 | 0.5 | 0.4527 | -0.0473 | 0.924 | 84035 | False |
| three_paradigm | social_reward | trust | 178 | 148 | 0.5618 | 0.5811 | 0.01928 | 0.924 | 84035 | False |
| three_paradigm | social_reward | socialdoors | 178 | 148 | 0.5281 | 0.5338 | 0.005694 | 0.924 | 84035 | False |
| partner_pair | closeness_positive | sharedreward | 192 | 165 | 0.7656 | 0.7636 | -0.001989 | 0.9455 | 84131 | False |
| partner_pair | closeness_positive | trust | 192 | 165 | 0.7812 | 0.7818 | 0.0005682 | 0.9455 | 84131 | False |
| partner_pair | closeness_negative | sharedreward | 192 | 165 | 0.8021 | 0.7879 | -0.0142 | 0.9405 | 84131 | False |
| partner_pair | closeness_negative | trust | 192 | 165 | 0.8385 | 0.8242 | -0.0143 | 0.9405 | 84131 | False |
| partner_pair | valence | sharedreward | 192 | 165 | 0.8646 | 0.8727 | 0.008144 | 0.9487 | 84131 | False |
| partner_pair | valence | trust | 192 | 165 | 0.9531 | 0.9515 | -0.00161 | 0.9487 | 84131 | False |

These compare already-completed models; there was no refitting for the QC comparison. Differences can reflect both sample and mask changes. Similarity across QC policies is descriptive robustness, not independent replication.

## Run-level individual-difference reliability

| cohort | family | task | n | pearson_r | spearman_rho | icc_3_1 |
| --- | --- | --- | --- | --- | --- | --- |
| three_paradigm | social_context | sharedreward | 173 | 0.1202 | 0.1125 | 0.1199 |
| three_paradigm | social_context | trust | 178 | 0.02391 | 0.0611 | 0.0239 |
| three_paradigm | social_reward | sharedreward | 173 | -0.008062 | -0.07103 | -0.00774 |
| three_paradigm | social_reward | trust | 178 | -0.1811 | -0.1888 | -0.1759 |
| partner_pair | closeness_positive | sharedreward | 186 | 0.3 | 0.2617 | 0.2985 |
| partner_pair | closeness_positive | trust | 192 | 0.2229 | 0.135 | 0.2212 |
| partner_pair | closeness_negative | sharedreward | 186 | 0.1321 | 0.1062 | 0.1321 |
| partner_pair | closeness_negative | trust | 192 | 0.08468 | 0.05537 | 0.08467 |
| partner_pair | closeness_reward | sharedreward | 186 | -0.001853 | -0.01214 | -0.001846 |
| partner_pair | closeness_reward | trust | 192 | 0.1019 | 0.1135 | 0.1013 |
| partner_pair | valence | sharedreward | 186 | 0.08915 | 0.06872 | 0.0866 |
| partner_pair | valence | trust | 192 | 0.03024 | 0.02169 | 0.03022 |

High condition-level forced-choice decoding and stable ranking of individuals are distinct properties. Low run-to-run margin correlations/ICC limit interpretation of a score as an individual-difference measure even when class ordering is accurate. These margins also come from fold-specific decoders; their score scaling and reliability should be considered before using a single score for between-person inference.

## Readiness for validation

The following observed development cells exceed their diagnostic permutation distributions at nominal p≤.05 with centered nulls; this is a candidate list for preregistration, not validation success:

| family | scope | test_task | observed_accuracy | empirical_p |
| --- | --- | --- | --- | --- |
| social_context | common | sharedreward | 0.9551 | 0.004975 |
| social_context | common | trust | 0.9382 | 0.004975 |
| social_context | common | socialdoors | 0.7753 | 0.004975 |
| social_context | lopo | sharedreward | 0.8202 | 0.004975 |
| social_context | lopo | trust | 0.9551 | 0.004975 |
| closeness_positive | common | sharedreward | 0.7656 | 0.004975 |
| closeness_positive | common | trust | 0.7812 | 0.004975 |
| closeness_negative | common | sharedreward | 0.8021 | 0.004975 |
| closeness_negative | common | trust | 0.8385 | 0.004975 |
| valence | common | sharedreward | 0.8646 | 0.004975 |
| valence | common | trust | 0.9531 | 0.004975 |

Prioritize condition discrimination claims supported by task-transfer, spatial review and sanity checks. Broad three-paradigm generalization requires inspecting the held-out Doors result specifically. Individual-difference claims require stronger reliability evidence. Select final constructs, transfer directions, endpoints and multiplicity before any later validation decision. No internal/external validation scoring or anatomical exclusion-mask analysis is implemented here.

## Figure package

- [Figure 1 PNG](../../../results/revised/characterization/figures/Figure1_schematic.png) · [PDF](../../../results/revised/characterization/figures/Figure1_schematic.pdf)
- [Figure 2 PNG](../../../results/revised/characterization/figures/Figure2_context_reward_transfer.png) · [PDF](../../../results/revised/characterization/figures/Figure2_context_reward_transfer.pdf)
- [Figure 3 PNG](../../../results/revised/characterization/figures/Figure3_social_context_pattern.png) · [PDF](../../../results/revised/characterization/figures/Figure3_social_context_pattern.pdf)
- [Figure 4 PNG](../../../results/revised/characterization/figures/Figure4_closeness_valence.png) · [PDF](../../../results/revised/characterization/figures/Figure4_closeness_valence.pdf)
- [Figure 5 PNG](../../../results/revised/characterization/figures/Figure5_oof_expression.png) · [PDF](../../../results/revised/characterization/figures/Figure5_oof_expression.pdf)
- [Figure 6 PNG](../../../results/revised/characterization/figures/Figure6_run_reliability.png) · [PDF](../../../results/revised/characterization/figures/Figure6_run_reliability.pdf)

Supplemental figures include OOF margins, QC robustness, all permutation nulls, raw/Haufe/mean maps and task/common spatial similarities. Charts retain vector text in PDF/SVG; PNG files use 300 dpi.

Original v4 hashes are checked again after plotting. Check `provenance/revised/characterization/run_status.json` for final completion.

**holdout_scored = False. Validation N=50 remains untouched.**
