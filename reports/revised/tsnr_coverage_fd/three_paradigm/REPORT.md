# Revised social reward, context, and closeness development analysis

| policy | cohort | development_n | original_development_retained_n | new_development_n | original_holdout_total_n | protected_holdout_total_n | supplemental_holdout_n | qc_qualified_holdout_n | original_holdout_unavailable_n | holdout_scored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tsnr_coverage_fd | three_paradigm | 178 | 166 | 12 | 50 | 62 | 12 | 41 | 14 | False |

All original holdouts remain barred from training; no validation images were scored.

## social_reward

Mean human reward-minus-negative-outcome map versus computer/monetary counterpart.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 178 | 0.5 | 0.5 | 0.4243 | 0.5757 | 0.0001441 | 1 | 1 |
| sharedreward | trust | pairwise | 178 | 0.5 | 0.573 | 0.4969 | 0.6467 | 0.1671 | 0.06065 | 0.9097 |
| sharedreward | socialdoors | pairwise | 178 | 0.5 | 0.4607 | 0.3858 | 0.5368 | -0.1571 | 0.3299 | 1 |
| trust | sharedreward | pairwise | 178 | 0.5 | 0.5393 | 0.4632 | 0.6142 | 0.02639 | 0.3299 | 1 |
| trust | trust | pairwise | 178 | 0.5 | 0.5337 | 0.4576 | 0.6087 | 0.04166 | 0.4097 | 1 |
| trust | socialdoors | pairwise | 178 | 0.5 | 0.5393 | 0.4632 | 0.6142 | 0.1274 | 0.3299 | 1 |
| socialdoors | sharedreward | pairwise | 178 | 0.5 | 0.4663 | 0.3913 | 0.5424 | -0.0121 | 0.4097 | 1 |
| socialdoors | trust | pairwise | 178 | 0.5 | 0.5281 | 0.452 | 0.6032 | 0.03122 | 0.5001 | 1 |
| socialdoors | socialdoors | pairwise | 178 | 0.5 | 0.5506 | 0.4744 | 0.6251 | 0.09689 | 0.2025 | 1 |
| trust+socialdoors | sharedreward | lopo | 178 | 0.5 | 0.5056 | 0.4298 | 0.5812 | 0.008852 | 0.9403 | 1 |
| sharedreward+socialdoors | trust | lopo | 178 | 0.5 | 0.5506 | 0.4744 | 0.6251 | 0.09143 | 0.2025 | 1 |
| sharedreward+trust | socialdoors | lopo | 178 | 0.5 | 0.5169 | 0.4409 | 0.5922 | -0.01606 | 0.7079 | 1 |
| sharedreward+trust+socialdoors | sharedreward | common | 178 | 0.5 | 0.5 | 0.4243 | 0.5757 | -0.01471 | 1 | 1 |
| sharedreward+trust+socialdoors | trust | common | 178 | 0.5 | 0.5618 | 0.4856 | 0.6359 | 0.09666 | 0.1152 | 1 |
| sharedreward+trust+socialdoors | socialdoors | common | 178 | 0.5 | 0.5281 | 0.452 | 0.6032 | 0.07712 | 0.5001 | 1 |

## social_context

Human versus computer/monetary context, averaging positive and negative outcomes equally.

| train_tasks | test_task | scope | n | chance | accuracy | ci_low | ci_high | mean_margin | binomial_p | holm_p_binary_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward | sharedreward | pairwise | 178 | 0.5 | 0.9607 | 0.9207 | 0.984 | 1.147 | 5.423e-42 | 8.135e-41 |
| sharedreward | trust | pairwise | 178 | 0.5 | 0.9607 | 0.9207 | 0.984 | 2.183 | 5.423e-42 | 8.135e-41 |
| sharedreward | socialdoors | pairwise | 178 | 0.5 | 0.4607 | 0.3858 | 0.5368 | -0.688 | 0.3299 | 0.6598 |
| trust | sharedreward | pairwise | 178 | 0.5 | 0.8876 | 0.8318 | 0.93 | 0.3837 | 8.248e-28 | 6.599e-27 |
| trust | trust | pairwise | 178 | 0.5 | 0.9382 | 0.8921 | 0.9688 | 1.359 | 5.8e-37 | 5.8e-36 |
| trust | socialdoors | pairwise | 178 | 0.5 | 0.7191 | 0.647 | 0.7838 | 1.395 | 4.485e-09 | 1.794e-08 |
| socialdoors | sharedreward | pairwise | 178 | 0.5 | 0.6854 | 0.6117 | 0.7528 | 0.03403 | 8.372e-07 | 2.512e-06 |
| socialdoors | trust | pairwise | 178 | 0.5 | 0.8427 | 0.7807 | 0.8929 | 0.1864 | 2.289e-21 | 1.602e-20 |
| socialdoors | socialdoors | pairwise | 178 | 0.5 | 0.9438 | 0.8991 | 0.9727 | 1.332 | 3.773e-38 | 4.15e-37 |
| trust+socialdoors | sharedreward | lopo | 178 | 0.5 | 0.8202 | 0.7558 | 0.8737 | 0.3186 | 1.342e-18 | 8.051e-18 |
| sharedreward+socialdoors | trust | lopo | 178 | 0.5 | 0.9551 | 0.9134 | 0.9804 | 2.156 | 1.167e-40 | 1.517e-39 |
| sharedreward+trust | socialdoors | lopo | 178 | 0.5 | 0.4888 | 0.4133 | 0.5647 | -0.2618 | 0.8222 | 0.8222 |
| sharedreward+trust+socialdoors | sharedreward | common | 178 | 0.5 | 0.9551 | 0.9134 | 0.9804 | 0.9886 | 1.167e-40 | 1.517e-39 |
| sharedreward+trust+socialdoors | trust | common | 178 | 0.5 | 0.9382 | 0.8921 | 0.9688 | 2.177 | 5.8e-37 | 5.8e-36 |
| sharedreward+trust+socialdoors | socialdoors | common | 178 | 0.5 | 0.7753 | 0.7068 | 0.8343 | 3.482 | 8.056e-14 | 4.028e-13 |

## Repeated-run reliability

| family | task | n | pearson_r | spearman_rho | icc_3_1 |
| --- | --- | --- | --- | --- | --- |
| social_context | sharedreward | 173 | 0.1202 | 0.1125 | 0.1199 |
| social_context | trust | 178 | 0.02391 | 0.0611 | 0.0239 |
| social_reward | sharedreward | 173 | -0.008062 | -0.07103 | -0.00774 |
| social_reward | trust | 178 | -0.1811 | -0.1888 | -0.1759 |

## Interpretation limits

- Binary accuracy is paired map ranking per participant, with ties incorrect.
- Three-class accuracy averages three map classifications within each participant; intervals bootstrap participants, not maps. No independent-map binomial p value is reported.
- Binary p values and intervals are descriptive CV summaries; Holm correction is within each family and sample/policy, not across the complete project. Overlapping training folds are not independent replications.
- Shared Reward is full-trial; Trust is outcome-phase. Averaging outcome valence does not isolate a pure social mechanism.
- The same participant is excluded from training across every task and condition. LOPO additionally excludes the test paradigm.
- Fixed masks use development-only coverage. Differences between policies/cohorts can reflect both sample and mask changes.
- Newly eligible development participants enter only if explicitly enabled; preserved original fold assignments and deterministic new folds apply across policies and cohorts.
- No UGR data or decision-phase maps are used. No settings are tuned to the revised results.
