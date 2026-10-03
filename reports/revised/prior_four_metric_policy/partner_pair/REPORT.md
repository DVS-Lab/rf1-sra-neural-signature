# Revised social reward, context, and closeness development analysis

| policy | cohort | development_n | original_development_retained_n | new_development_n | original_holdout_total_n | protected_holdout_total_n | supplemental_holdout_n | qc_qualified_holdout_n | original_holdout_unavailable_n | holdout_scored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prior_four_metric_policy | partner_pair | 165 | 152 | 13 | 50 | 62 | 12 | 44 | 17 | False |

All original holdouts remain barred from training; no validation images were scored.

## partner_reward

Computer / friend / stranger identity from reward-minus-negative-outcome maps.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 165 | 0.3333 | 0.3293 | 0.2889 | 0.3697 | -0.2577 | NA | NA |
| sharedreward | trust | pairwise | 165 | 0.3333 | 0.3313 | 0.2909 | 0.3717 | -0.5579 | NA | NA |
| trust | sharedreward | pairwise | 165 | 0.3333 | 0.3495 | 0.3091 | 0.3879 | -0.115 | NA | NA |
| trust | trust | pairwise | 165 | 0.3333 | 0.3414 | 0.3071 | 0.3778 | -0.2531 | NA | NA |
| sharedreward+trust | sharedreward | common | 165 | 0.3333 | 0.3192 | 0.2808 | 0.3576 | -0.254 | NA | NA |
| sharedreward+trust | trust | common | 165 | 0.3333 | 0.3374 | 0.2949 | 0.3798 | -0.5253 | NA | NA |

## social_reward

Mean human reward-minus-negative-outcome map versus computer/monetary counterpart.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 165 | 0.5 | 0.4727 | 0.3946 | 0.5518 | -0.0374 | 0.5335 | 1 |
| sharedreward | trust | pairwise | 165 | 0.5 | 0.5697 | 0.4904 | 0.6464 | 0.2003 | 0.08646 | 0.5188 |
| trust | sharedreward | pairwise | 165 | 0.5 | 0.5697 | 0.4904 | 0.6464 | 0.03587 | 0.08646 | 0.5188 |
| trust | trust | pairwise | 165 | 0.5 | 0.5636 | 0.4844 | 0.6406 | 0.0579 | 0.1192 | 0.5188 |
| sharedreward+trust | sharedreward | common | 165 | 0.5 | 0.4788 | 0.4005 | 0.5578 | -0.03025 | 0.6406 | 1 |
| sharedreward+trust | trust | common | 165 | 0.5 | 0.5636 | 0.4844 | 0.6406 | 0.1388 | 0.1192 | 0.5188 |

## social_context

Human versus computer/monetary context, averaging positive and negative outcomes equally.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 165 | 0.5 | 0.9576 | 0.9145 | 0.9828 | 1.123 | 2.597e-38 | 1.298e-37 |
| sharedreward | trust | pairwise | 165 | 0.5 | 0.9576 | 0.9145 | 0.9828 | 2.167 | 2.597e-38 | 1.298e-37 |
| trust | sharedreward | pairwise | 165 | 0.5 | 0.8667 | 0.8051 | 0.9145 | 0.3749 | 6.309e-23 | 6.309e-23 |
| trust | trust | pairwise | 165 | 0.5 | 0.9636 | 0.9225 | 0.9865 | 1.373 | 1.136e-39 | 6.813e-39 |
| sharedreward+trust | sharedreward | common | 165 | 0.5 | 0.9515 | 0.9067 | 0.9788 | 1.085 | 5.164e-37 | 1.549e-36 |
| sharedreward+trust | trust | common | 165 | 0.5 | 0.9515 | 0.9067 | 0.9788 | 2.264 | 5.164e-37 | 1.549e-36 |

## closeness_positive

Friend versus stranger during reward/reciprocation.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 165 | 0.5 | 0.7818 | 0.711 | 0.8423 | 0.5314 | 1.725e-13 | 8.625e-13 |
| sharedreward | trust | pairwise | 165 | 0.5 | 0.7515 | 0.6784 | 0.8154 | 0.9735 | 6.801e-11 | 1.36e-10 |
| trust | sharedreward | pairwise | 165 | 0.5 | 0.6727 | 0.5955 | 0.7436 | 0.1554 | 1.078e-05 | 1.078e-05 |
| trust | trust | pairwise | 165 | 0.5 | 0.8 | 0.7308 | 0.8581 | 0.6045 | 3.165e-15 | 1.899e-14 |
| sharedreward+trust | sharedreward | common | 165 | 0.5 | 0.7636 | 0.6914 | 0.8262 | 0.4522 | 6.875e-12 | 2.063e-11 |
| sharedreward+trust | trust | common | 165 | 0.5 | 0.7818 | 0.711 | 0.8423 | 1.046 | 1.725e-13 | 8.625e-13 |

## closeness_negative

Friend versus stranger during punishment/defection.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 165 | 0.5 | 0.8061 | 0.7374 | 0.8634 | 0.528 | 7.758e-16 | 2.327e-15 |
| sharedreward | trust | pairwise | 165 | 0.5 | 0.8121 | 0.744 | 0.8686 | 1.411 | 1.831e-16 | 7.324e-16 |
| trust | sharedreward | pairwise | 165 | 0.5 | 0.7576 | 0.6848 | 0.8208 | 0.2007 | 2.197e-11 | 2.197e-11 |
| trust | trust | pairwise | 165 | 0.5 | 0.8788 | 0.819 | 0.9244 | 0.9122 | 1.371e-24 | 8.225e-24 |
| sharedreward+trust | sharedreward | common | 165 | 0.5 | 0.7879 | 0.7175 | 0.8476 | 0.5099 | 4.715e-14 | 9.43e-14 |
| sharedreward+trust | trust | common | 165 | 0.5 | 0.8242 | 0.7574 | 0.879 | 1.457 | 9.068e-18 | 4.534e-17 |

## closeness_reward

Friend versus stranger reward-minus-negative-outcome maps.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 165 | 0.5 | 0.5212 | 0.4422 | 0.5995 | 0.009822 | 0.6406 | 1 |
| sharedreward | trust | pairwise | 165 | 0.5 | 0.4909 | 0.4124 | 0.5698 | -0.02586 | 0.8763 | 1 |
| trust | sharedreward | pairwise | 165 | 0.5 | 0.4424 | 0.3653 | 0.5217 | -0.02597 | 0.1609 | 0.8046 |
| trust | trust | pairwise | 165 | 0.5 | 0.5636 | 0.4844 | 0.6406 | 0.1069 | 0.1192 | 0.7152 |
| sharedreward+trust | sharedreward | common | 165 | 0.5 | 0.4788 | 0.4005 | 0.5578 | -0.02388 | 0.6406 | 1 |
| sharedreward+trust | trust | common | 165 | 0.5 | 0.5152 | 0.4362 | 0.5936 | 0.02761 | 0.7556 | 1 |

## valence

Positive versus negative outcome maps, averaging computer/friend/stranger equally.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 165 | 0.5 | 0.8424 | 0.7777 | 0.8944 | 0.4919 | 7.329e-20 | 1.466e-19 |
| sharedreward | trust | pairwise | 165 | 0.5 | 0.9212 | 0.869 | 0.9574 | 1.79 | 3.102e-31 | 1.241e-30 |
| trust | sharedreward | pairwise | 165 | 0.5 | 0.7939 | 0.7241 | 0.8529 | 0.1863 | 1.244e-14 | 1.244e-14 |
| trust | trust | pairwise | 165 | 0.5 | 0.9515 | 0.9067 | 0.9788 | 1.125 | 5.164e-37 | 3.098e-36 |
| sharedreward+trust | sharedreward | common | 165 | 0.5 | 0.8727 | 0.8121 | 0.9195 | 0.5215 | 9.55e-24 | 2.865e-23 |
| sharedreward+trust | trust | common | 165 | 0.5 | 0.9515 | 0.9067 | 0.9788 | 2.022 | 5.164e-37 | 3.098e-36 |

## Repeated-run reliability

| family | task | n | pearson_r | spearman_rho | icc_3_1 |
| --- | --- | --- | --- | --- | --- |
| closeness_negative | sharedreward | 160 | 0.205 | 0.153 | 0.205 |
| closeness_negative | trust | 165 | 0.03923 | 0.05215 | 0.03919 |
| closeness_positive | sharedreward | 160 | 0.3291 | 0.2642 | 0.3258 |
| closeness_positive | trust | 165 | 0.1766 | 0.09587 | 0.1753 |
| closeness_reward | sharedreward | 160 | -0.003702 | -0.01095 | -0.003701 |
| closeness_reward | trust | 165 | 0.1194 | 0.1281 | 0.1194 |
| social_context | sharedreward | 160 | 0.2808 | 0.2539 | 0.2807 |
| social_context | trust | 165 | 0.009382 | 0.03949 | 0.009376 |
| social_reward | sharedreward | 160 | -0.1001 | -0.1006 | -0.09749 |
| social_reward | trust | 165 | -0.06417 | -0.0916 | -0.06344 |
| valence | sharedreward | 160 | 0.06546 | 0.07118 | 0.06312 |
| valence | trust | 165 | 0.01802 | 0.02773 | 0.01801 |

## Interpretation limits

- Binary accuracy is paired map ranking per participant, with ties incorrect.
- Three-class accuracy averages three map classifications within each participant; intervals bootstrap participants, not maps. No independent-map binomial p value is reported.
- Binary p values and intervals are descriptive CV summaries; Holm correction is within each family and sample/policy, not across the complete project. Overlapping training folds are not independent replications.
- Shared Reward is full-trial; Trust is outcome-phase. Averaging outcome valence does not isolate a pure social mechanism.
- The same participant is excluded from training across every task and condition. LOPO additionally excludes the test paradigm.
- Fixed masks use development-only coverage. Differences between policies/cohorts can reflect both sample and mask changes.
- Newly eligible development participants enter only if explicitly enabled; preserved original fold assignments and deterministic new folds apply across policies and cohorts.
- No UGR data or decision-phase maps are used. No settings are tuned to the revised results.
