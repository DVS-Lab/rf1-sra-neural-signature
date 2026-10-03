# Revised social reward, context, and closeness development analysis

| policy | cohort | development_n | original_development_retained_n | new_development_n | original_holdout_total_n | protected_holdout_total_n | supplemental_holdout_n | qc_qualified_holdout_n | original_holdout_unavailable_n | holdout_scored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prior_four_metric_policy | three_paradigm | 148 | 139 | 9 | 50 | 62 | 12 | 36 | 19 | False |

All original holdouts remain barred from training; no validation images were scored.

## social_reward

Mean human reward-minus-negative-outcome map versus computer/monetary counterpart.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 148 | 0.5 | 0.5 | 0.4168 | 0.5832 | -0.0309 | 1 | 1 |
| sharedreward | trust | pairwise | 148 | 0.5 | 0.5608 | 0.4769 | 0.6422 | 0.2331 | 0.1621 | 1 |
| sharedreward | socialdoors | pairwise | 148 | 0.5 | 0.5068 | 0.4234 | 0.5898 | -0.01661 | 0.9345 | 1 |
| trust | sharedreward | pairwise | 148 | 0.5 | 0.6081 | 0.5246 | 0.6872 | 0.04328 | 0.01058 | 0.1587 |
| trust | trust | pairwise | 148 | 0.5 | 0.5203 | 0.4367 | 0.603 | 0.01792 | 0.6812 | 1 |
| trust | socialdoors | pairwise | 148 | 0.5 | 0.5743 | 0.4905 | 0.6551 | 0.1642 | 0.08397 | 1 |
| socialdoors | sharedreward | pairwise | 148 | 0.5 | 0.4932 | 0.4102 | 0.5766 | -0.006148 | 0.9345 | 1 |
| socialdoors | trust | pairwise | 148 | 0.5 | 0.5203 | 0.4367 | 0.603 | 0.04583 | 0.6812 | 1 |
| socialdoors | socialdoors | pairwise | 148 | 0.5 | 0.5541 | 0.4702 | 0.6357 | 0.1009 | 0.2174 | 1 |
| trust+socialdoors | sharedreward | lopo | 148 | 0.5 | 0.5338 | 0.4501 | 0.6161 | 0.02232 | 0.4595 | 1 |
| sharedreward+socialdoors | trust | lopo | 148 | 0.5 | 0.5203 | 0.4367 | 0.603 | 0.1926 | 0.6812 | 1 |
| sharedreward+trust | socialdoors | lopo | 148 | 0.5 | 0.5068 | 0.4234 | 0.5898 | 0.1107 | 0.9345 | 1 |
| sharedreward+trust+socialdoors | sharedreward | common | 148 | 0.5 | 0.4527 | 0.3708 | 0.5365 | -0.05548 | 0.2852 | 1 |
| sharedreward+trust+socialdoors | trust | common | 148 | 0.5 | 0.5811 | 0.4973 | 0.6616 | 0.1633 | 0.05831 | 0.8164 |
| sharedreward+trust+socialdoors | socialdoors | common | 148 | 0.5 | 0.5338 | 0.4501 | 0.6161 | 0.1991 | 0.4595 | 1 |

## social_context

Human versus computer/monetary context, averaging positive and negative outcomes equally.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 148 | 0.5 | 0.9527 | 0.905 | 0.9808 | 1.117 | 1.575e-33 | 1.732e-32 |
| sharedreward | trust | pairwise | 148 | 0.5 | 0.9595 | 0.9139 | 0.985 | 2.266 | 7.704e-35 | 1.079e-33 |
| sharedreward | socialdoors | pairwise | 148 | 0.5 | 0.5203 | 0.4367 | 0.603 | -0.1623 | 0.6812 | 0.6812 |
| trust | sharedreward | pairwise | 148 | 0.5 | 0.8784 | 0.8146 | 0.9263 | 0.3773 | 4.006e-22 | 3.205e-21 |
| trust | trust | pairwise | 148 | 0.5 | 0.9527 | 0.905 | 0.9808 | 1.349 | 1.575e-33 | 1.732e-32 |
| trust | socialdoors | pairwise | 148 | 0.5 | 0.7027 | 0.6221 | 0.775 | 1.518 | 8.873e-07 | 3.549e-06 |
| socialdoors | sharedreward | pairwise | 148 | 0.5 | 0.6959 | 0.615 | 0.7688 | 0.03448 | 2.097e-06 | 6.291e-06 |
| socialdoors | trust | pairwise | 148 | 0.5 | 0.8514 | 0.7836 | 0.9044 | 0.1781 | 6.485e-19 | 3.891e-18 |
| socialdoors | socialdoors | pairwise | 148 | 0.5 | 0.9662 | 0.9229 | 0.9889 | 1.409 | 3.209e-36 | 4.813e-35 |
| trust+socialdoors | sharedreward | lopo | 148 | 0.5 | 0.8649 | 0.799 | 0.9155 | 0.3137 | 1.804e-20 | 1.263e-19 |
| sharedreward+socialdoors | trust | lopo | 148 | 0.5 | 0.9595 | 0.9139 | 0.985 | 2.222 | 7.704e-35 | 1.079e-33 |
| sharedreward+trust | socialdoors | lopo | 148 | 0.5 | 0.5608 | 0.4769 | 0.6422 | 0.3898 | 0.1621 | 0.3241 |
| sharedreward+trust+socialdoors | sharedreward | common | 148 | 0.5 | 0.9595 | 0.9139 | 0.985 | 0.9643 | 7.704e-35 | 1.079e-33 |
| sharedreward+trust+socialdoors | trust | common | 148 | 0.5 | 0.9459 | 0.8963 | 0.9764 | 2.248 | 2.797e-32 | 2.518e-31 |
| sharedreward+trust+socialdoors | socialdoors | common | 148 | 0.5 | 0.8108 | 0.7383 | 0.8705 | 3.986 | 9.106e-15 | 4.553e-14 |

## Repeated-run reliability

| family | task | n | pearson_r | spearman_rho | icc_3_1 |
| --- | --- | --- | --- | --- | --- |
| social_context | sharedreward | 144 | 0.1731 | 0.1807 | 0.172 |
| social_context | trust | 148 | 0.02097 | 0.04713 | 0.02092 |
| social_reward | sharedreward | 144 | -0.05074 | -0.08487 | -0.04944 |
| social_reward | trust | 148 | -0.1071 | -0.08067 | -0.1042 |

## Interpretation limits

- Binary accuracy is paired map ranking per participant, with ties incorrect.
- Three-class accuracy averages three map classifications within each participant; intervals bootstrap participants, not maps. No independent-map binomial p value is reported.
- Binary p values and intervals are descriptive CV summaries; Holm correction is within each family and sample/policy, not across the complete project. Overlapping training folds are not independent replications.
- Shared Reward is full-trial; Trust is outcome-phase. Averaging outcome valence does not isolate a pure social mechanism.
- The same participant is excluded from training across every task and condition. LOPO additionally excludes the test paradigm.
- Fixed masks use development-only coverage. Differences between policies/cohorts can reflect both sample and mask changes.
- Newly eligible development participants enter only if explicitly enabled; preserved original fold assignments and deterministic new folds apply across policies and cohorts.
- No UGR data or decision-phase maps are used. No settings are tuned to the revised results.
