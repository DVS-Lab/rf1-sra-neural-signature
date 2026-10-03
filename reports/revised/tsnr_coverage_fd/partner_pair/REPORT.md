# Revised social reward, context, and closeness development analysis

| policy | cohort | development_n | original_development_retained_n | new_development_n | original_holdout_total_n | protected_holdout_total_n | supplemental_holdout_n | qc_qualified_holdout_n | original_holdout_unavailable_n | holdout_scored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tsnr_coverage_fd | partner_pair | 192 | 176 | 16 | 50 | 62 | 12 | 50 | 12 | False |

All original holdouts remain barred from training; no validation images were scored.

## partner_reward

Computer / friend / stranger identity from reward-minus-negative-outcome maps.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 192 | 0.3333 | 0.349 | 0.3108 | 0.3889 | -0.2379 | NA | NA |
| sharedreward | trust | pairwise | 192 | 0.3333 | 0.3368 | 0.2986 | 0.375 | -0.5824 | NA | NA |
| trust | sharedreward | pairwise | 192 | 0.3333 | 0.3542 | 0.316 | 0.3924 | -0.1194 | NA | NA |
| trust | trust | pairwise | 192 | 0.3333 | 0.3351 | 0.3003 | 0.3698 | -0.2784 | NA | NA |
| sharedreward+trust | sharedreward | common | 192 | 0.3333 | 0.3333 | 0.2951 | 0.3715 | -0.244 | NA | NA |
| sharedreward+trust | trust | common | 192 | 0.3333 | 0.3438 | 0.3038 | 0.3819 | -0.5725 | NA | NA |

## social_reward

Mean human reward-minus-negative-outcome map versus computer/monetary counterpart.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 192 | 0.5 | 0.4948 | 0.422 | 0.5677 | -0.006155 | 0.9425 | 1 |
| sharedreward | trust | pairwise | 192 | 0.5 | 0.5312 | 0.4581 | 0.6035 | 0.1185 | 0.4274 | 1 |
| trust | sharedreward | pairwise | 192 | 0.5 | 0.5 | 0.4272 | 0.5728 | 0.01764 | 1 | 1 |
| trust | trust | pairwise | 192 | 0.5 | 0.5312 | 0.4581 | 0.6035 | 0.0547 | 0.4274 | 1 |
| sharedreward+trust | sharedreward | common | 192 | 0.5 | 0.4896 | 0.4169 | 0.5626 | 0.001237 | 0.8287 | 1 |
| sharedreward+trust | trust | common | 192 | 0.5 | 0.5521 | 0.4788 | 0.6237 | 0.08255 | 0.1701 | 1 |

## social_context

Human versus computer/monetary context, averaging positive and negative outcomes equally.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 192 | 0.5 | 0.9583 | 0.9196 | 0.9818 | 1.16 | 1.315e-44 | 6.577e-44 |
| sharedreward | trust | pairwise | 192 | 0.5 | 0.9375 | 0.8934 | 0.9673 | 2.143 | 1.258e-39 | 2.517e-39 |
| trust | sharedreward | pairwise | 192 | 0.5 | 0.8854 | 0.8317 | 0.9268 | 0.3898 | 1.59e-29 | 1.59e-29 |
| trust | trust | pairwise | 192 | 0.5 | 0.9427 | 0.8998 | 0.9711 | 1.39 | 8.291e-41 | 2.487e-40 |
| sharedreward+trust | sharedreward | common | 192 | 0.5 | 0.9635 | 0.9263 | 0.9852 | 1.13 | 5.655e-46 | 3.393e-45 |
| sharedreward+trust | trust | common | 192 | 0.5 | 0.9531 | 0.9129 | 0.9783 | 2.258 | 2.705e-43 | 1.082e-42 |

## closeness_positive

Friend versus stranger during reward/reciprocation.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 192 | 0.5 | 0.776 | 0.7104 | 0.8329 | 0.5398 | 6.859e-15 | 2.744e-14 |
| sharedreward | trust | pairwise | 192 | 0.5 | 0.7552 | 0.6881 | 0.8143 | 1.099 | 7.966e-13 | 1.593e-12 |
| trust | sharedreward | pairwise | 192 | 0.5 | 0.6771 | 0.606 | 0.7426 | 0.1622 | 1.037e-06 | 1.037e-06 |
| trust | trust | pairwise | 192 | 0.5 | 0.8073 | 0.7443 | 0.8605 | 0.5922 | 2.242e-18 | 1.345e-17 |
| sharedreward+trust | sharedreward | common | 192 | 0.5 | 0.7656 | 0.6992 | 0.8236 | 0.4673 | 7.825e-14 | 2.347e-13 |
| sharedreward+trust | trust | common | 192 | 0.5 | 0.7812 | 0.716 | 0.8376 | 1.112 | 1.944e-15 | 9.718e-15 |

## closeness_negative

Friend versus stranger during punishment/defection.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 192 | 0.5 | 0.8177 | 0.7557 | 0.8696 | 0.528 | 1.195e-19 | 4.782e-19 |
| sharedreward | trust | pairwise | 192 | 0.5 | 0.8177 | 0.7557 | 0.8696 | 1.405 | 1.195e-19 | 4.782e-19 |
| trust | sharedreward | pairwise | 192 | 0.5 | 0.75 | 0.6826 | 0.8096 | 0.2115 | 2.438e-12 | 2.438e-12 |
| trust | trust | pairwise | 192 | 0.5 | 0.8802 | 0.8257 | 0.9225 | 0.9769 | 1.184e-28 | 7.103e-28 |
| sharedreward+trust | sharedreward | common | 192 | 0.5 | 0.8021 | 0.7386 | 0.856 | 0.5073 | 9.242e-18 | 1.848e-17 |
| sharedreward+trust | trust | common | 192 | 0.5 | 0.8385 | 0.7787 | 0.8876 | 1.501 | 2.236e-22 | 1.118e-21 |

## closeness_reward

Friend versus stranger reward-minus-negative-outcome maps.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 192 | 0.5 | 0.5417 | 0.4684 | 0.6136 | 0.04755 | 0.279 | 1 |
| sharedreward | trust | pairwise | 192 | 0.5 | 0.5 | 0.4272 | 0.5728 | 0.04544 | 1 | 1 |
| trust | sharedreward | pairwise | 192 | 0.5 | 0.4948 | 0.422 | 0.5677 | -0.005218 | 0.9425 | 1 |
| trust | trust | pairwise | 192 | 0.5 | 0.5677 | 0.4944 | 0.6388 | 0.08162 | 0.07092 | 0.4255 |
| sharedreward+trust | sharedreward | common | 192 | 0.5 | 0.5365 | 0.4632 | 0.6085 | 0.02432 | 0.3482 | 1 |
| sharedreward+trust | trust | common | 192 | 0.5 | 0.5052 | 0.4323 | 0.578 | 0.04203 | 0.9425 | 1 |

## valence

Positive versus negative outcome maps, averaging computer/friend/stranger equally.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 192 | 0.5 | 0.8333 | 0.7729 | 0.8831 | 0.5281 | 1.135e-21 | 2.271e-21 |
| sharedreward | trust | pairwise | 192 | 0.5 | 0.8958 | 0.8437 | 0.9352 | 1.742 | 2.46e-31 | 9.842e-31 |
| trust | sharedreward | pairwise | 192 | 0.5 | 0.776 | 0.7104 | 0.8329 | 0.1865 | 6.859e-15 | 6.859e-15 |
| trust | trust | pairwise | 192 | 0.5 | 0.9479 | 0.9063 | 0.9747 | 1.134 | 4.98e-42 | 2.49e-41 |
| sharedreward+trust | sharedreward | common | 192 | 0.5 | 0.8646 | 0.8079 | 0.9096 | 0.5571 | 3.684e-26 | 1.105e-25 |
| sharedreward+trust | trust | common | 192 | 0.5 | 0.9531 | 0.9129 | 0.9783 | 2.024 | 2.705e-43 | 1.623e-42 |

## Repeated-run reliability

| family | task | n | pearson_r | spearman_rho | icc_3_1 |
| --- | --- | --- | --- | --- | --- |
| closeness_negative | sharedreward | 186 | 0.1321 | 0.1062 | 0.1321 |
| closeness_negative | trust | 192 | 0.08468 | 0.05537 | 0.08467 |
| closeness_positive | sharedreward | 186 | 0.3 | 0.2617 | 0.2985 |
| closeness_positive | trust | 192 | 0.2229 | 0.135 | 0.2212 |
| closeness_reward | sharedreward | 186 | -0.001853 | -0.01214 | -0.001846 |
| closeness_reward | trust | 192 | 0.1019 | 0.1135 | 0.1013 |
| social_context | sharedreward | 186 | 0.2342 | 0.2068 | 0.2342 |
| social_context | trust | 192 | 0.02612 | 0.03837 | 0.02612 |
| social_reward | sharedreward | 186 | 0.0005698 | -0.02994 | 0.0005528 |
| social_reward | trust | 192 | -0.1062 | -0.1092 | -0.1049 |
| valence | sharedreward | 186 | 0.08915 | 0.06872 | 0.0866 |
| valence | trust | 192 | 0.03024 | 0.02169 | 0.03022 |

## Interpretation limits

- Binary accuracy is paired map ranking per participant, with ties incorrect.
- Three-class accuracy averages three map classifications within each participant; intervals bootstrap participants, not maps. No independent-map binomial p value is reported.
- Binary p values and intervals are descriptive CV summaries; Holm correction is within each family and sample/policy, not across the complete project. Overlapping training folds are not independent replications.
- Shared Reward is full-trial; Trust is outcome-phase. Averaging outcome valence does not isolate a pure social mechanism.
- The same participant is excluded from training across every task and condition. LOPO additionally excludes the test paradigm.
- Fixed masks use development-only coverage. Differences between policies/cohorts can reflect both sample and mask changes.
- Newly eligible development participants enter only if explicitly enabled; preserved original fold assignments and deterministic new folds apply across policies and cohorts.
- No UGR data or decision-phase maps are used. No settings are tuned to the revised results.
