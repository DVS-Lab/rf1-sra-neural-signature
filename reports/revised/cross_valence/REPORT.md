# Final development follow-up: cross-valence context

Development only. Participants, five folds, QC, masks, source images, centering and LinearSVC settings remain fixed. Partner-pair N=192. Three-paradigm N=178. Primary validation N=50 remains untouched. See [audit](IMPLEMENTATION_AUDIT.md).

## Social context: does it generalize across valence and across both task and valence?

Same task, cross valence: observed accuracy ranges from 85.9% to 89.6%; 4/4 descriptive 95% intervals lie entirely above chance. These intervals summarize OOF participants, not independent training-fold replications. Assess all directions; asymmetry limits invariance claims.

| train_domain | test_domain | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward_positive | sharedreward_negative | 192 | 0.8958 | 0.8437 | 0.9352 | 0.8657 | 0.7596 | 0.9771 |
| sharedreward_negative | sharedreward_positive | 192 | 0.8906 | 0.8377 | 0.931 | 0.7929 | 0.7022 | 0.8845 |
| trust_positive | trust_negative | 192 | 0.8802 | 0.8257 | 0.9225 | 1.121 | 0.9919 | 1.246 |
| trust_negative | trust_positive | 192 | 0.8594 | 0.802 | 0.9052 | 0.934 | 0.8163 | 1.05 |

Both task and valence change: observed accuracy ranges from 74.5% to 84.9%; 4/4 descriptive 95% intervals lie entirely above chance. These intervals summarize OOF participants, not independent training-fold replications. Assess all directions; asymmetry limits invariance claims.

| train_domain | test_domain | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward_positive | trust_negative | 192 | 0.8385 | 0.7787 | 0.8876 | 1.705 | 1.473 | 1.939 |
| sharedreward_negative | trust_positive | 192 | 0.849 | 0.7903 | 0.8965 | 1.276 | 1.07 | 1.485 |
| trust_positive | sharedreward_negative | 192 | 0.7448 | 0.677 | 0.8048 | 0.2598 | 0.2046 | 0.3147 |
| trust_negative | sharedreward_positive | 192 | 0.7917 | 0.7273 | 0.8468 | 0.2727 | 0.2255 | 0.3201 |

Training pooled across SR and Trust at one valence, testing exclusively at the other:

| train_domain | test_domain | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled_positive | sharedreward_negative | 192 | 0.9062 | 0.8559 | 0.9435 | 0.85 | 0.7441 | 0.9572 |
| pooled_positive | trust_negative | 192 | 0.875 | 0.8197 | 0.9182 | 1.87 | 1.647 | 2.092 |
| pooled_positive | combined | 192 | 0.8906 | 0.8568 | 0.9219 | 1.36 | 1.232 | 1.487 |
| pooled_negative | sharedreward_positive | 192 | 0.8854 | 0.8317 | 0.9268 | 0.7797 | 0.6899 | 0.87 |
| pooled_negative | trust_positive | 192 | 0.8698 | 0.8138 | 0.9139 | 1.442 | 1.248 | 1.643 |
| pooled_negative | combined | 192 | 0.8776 | 0.8438 | 0.9115 | 1.111 | 1 | 1.223 |

## Friend–stranger context: does it generalize across valence and across both task and valence?

Same task, cross valence: observed accuracy ranges from 76.0% to 87.5%; 4/4 descriptive 95% intervals lie entirely above chance. These intervals summarize OOF participants, not independent training-fold replications. Assess all directions; asymmetry limits invariance claims.

| train_domain | test_domain | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward_positive | sharedreward_negative | 192 | 0.7604 | 0.6937 | 0.8189 | 0.5295 | 0.4231 | 0.6354 |
| sharedreward_negative | sharedreward_positive | 192 | 0.7656 | 0.6992 | 0.8236 | 0.5244 | 0.4197 | 0.6328 |
| trust_positive | trust_negative | 192 | 0.875 | 0.8197 | 0.9182 | 0.8546 | 0.7332 | 0.9755 |
| trust_negative | trust_positive | 192 | 0.8281 | 0.7672 | 0.8786 | 0.6559 | 0.5475 | 0.7596 |

Both task and valence change: observed accuracy ranges from 67.2% to 84.9%; 4/4 descriptive 95% intervals lie entirely above chance. These intervals summarize OOF participants, not independent training-fold replications. Assess all directions; asymmetry limits invariance claims.

| train_domain | test_domain | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward_positive | trust_negative | 192 | 0.849 | 0.7903 | 0.8965 | 1.543 | 1.303 | 1.779 |
| sharedreward_negative | trust_positive | 192 | 0.7656 | 0.6992 | 0.8236 | 0.9563 | 0.759 | 1.155 |
| trust_positive | sharedreward_negative | 192 | 0.6719 | 0.6006 | 0.7378 | 0.149 | 0.1055 | 0.194 |
| trust_negative | sharedreward_positive | 192 | 0.6979 | 0.6277 | 0.7619 | 0.2038 | 0.1499 | 0.2596 |

Training pooled across SR and Trust at one valence, testing exclusively at the other:

| train_domain | test_domain | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled_positive | sharedreward_negative | 192 | 0.8229 | 0.7614 | 0.8741 | 0.4826 | 0.3932 | 0.5746 |
| pooled_positive | trust_negative | 192 | 0.849 | 0.7903 | 0.8965 | 1.511 | 1.293 | 1.725 |
| pooled_positive | combined | 192 | 0.8359 | 0.7969 | 0.8724 | 0.9968 | 0.8756 | 1.122 |
| pooled_negative | sharedreward_positive | 192 | 0.7969 | 0.733 | 0.8514 | 0.5085 | 0.4096 | 0.6105 |
| pooled_negative | trust_positive | 192 | 0.8281 | 0.7672 | 0.8786 | 1.1 | 0.9167 | 1.282 |
| pooled_negative | combined | 192 | 0.8125 | 0.7708 | 0.8516 | 0.804 | 0.7023 | 0.9102 |

## Does explicit valence-collapsed friend–stranger context classify both tasks?

| train_domain | test_domain | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| collapsed_sharedreward | sharedreward | 192 | 0.8438 | 0.7845 | 0.892 | 0.8001 | 0.6841 | 0.9181 |
| collapsed_sharedreward | trust | 192 | 0.8854 | 0.8317 | 0.9268 | 1.907 | 1.669 | 2.146 |
| collapsed_trust | sharedreward | 192 | 0.8177 | 0.7557 | 0.8696 | 0.2879 | 0.235 | 0.3428 |
| collapsed_trust | trust | 192 | 0.9323 | 0.887 | 0.9635 | 1.179 | 1.05 | 1.306 |
| collapsed_common | sharedreward | 192 | 0.8542 | 0.7962 | 0.9009 | 0.7553 | 0.6471 | 0.8666 |
| collapsed_common | trust | 192 | 0.875 | 0.8197 | 0.9182 | 2.028 | 1.794 | 2.263 |

This model averages outcomes before centering, then distinguishes friend from stranger. It may encode familiarity, identity, relationship status, visual or relationship-linked semantic information; it is not labeled pure closeness.

## Are positive, negative and collapsed spatial patterns similar?

| family | pattern_a | pattern_b | kind | spatial_r | n_voxels |
| --- | --- | --- | --- | --- | --- |
| social_context | positive | negative | weight | 0.2318 | 84133 |
| social_context | positive | negative | haufe | 0.889 | 84133 |
| social_context | positive | negative | mean_difference | 0.8768 | 84133 |
| social_context | positive | collapsed | weight | 0.7225 | 84133 |
| social_context | positive | collapsed | haufe | 0.9647 | 84133 |
| social_context | positive | collapsed | mean_difference | 0.9671 | 84133 |
| social_context | negative | collapsed | weight | 0.7339 | 84133 |
| social_context | negative | collapsed | haufe | 0.9662 | 84133 |
| social_context | negative | collapsed | mean_difference | 0.9702 | 84133 |
| friend_stranger_context | positive | negative | weight | 0.1405 | 84133 |
| friend_stranger_context | positive | negative | haufe | 0.833 | 84133 |
| friend_stranger_context | positive | negative | mean_difference | 0.8209 | 84133 |
| friend_stranger_context | positive | collapsed | weight | 0.6916 | 84133 |
| friend_stranger_context | positive | collapsed | haufe | 0.9392 | 84133 |
| friend_stranger_context | positive | collapsed | mean_difference | 0.9379 | 84133 |
| friend_stranger_context | negative | collapsed | weight | 0.6879 | 84133 |
| friend_stranger_context | negative | collapsed | haufe | 0.9591 | 84133 |
| friend_stranger_context | negative | collapsed | mean_difference | 0.968 | 84133 |

These spatial correlations describe patterns from overlapping development data and do not themselves demonstrate prediction or independent replication. Haufe patterns are Cov(X,score)/Var(score), without feature z-scoring. Raw weights are discriminative coefficients, not regional activation or necessity. All anatomical displays are descriptive and use fixed x=0, y=-50, z=20 mm cuts in the supplements.

Standard display background: `/ZPOOL/data/tools/templateflow/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz`; SHA256 `2a814da50173599a857d96246dc057d548072bd6dffa499f75724dbad20792b1`. FSL MNI backgrounds/atlases are descriptive display/label references and need not match the exact nonlinear template variant. No atlas or background was downloaded.

## Main effects versus the existing interactions

| family | test_task | n | accuracy | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| social_reward | sharedreward | 192 | 0.4896 | 0.4169 | 0.5626 |
| social_reward | trust | 192 | 0.5521 | 0.4788 | 0.6237 |
| social_context | sharedreward | 192 | 0.9635 | 0.9263 | 0.9852 |
| social_context | trust | 192 | 0.9531 | 0.9129 | 0.9783 |
| closeness_reward | sharedreward | 192 | 0.5365 | 0.4632 | 0.6085 |
| closeness_reward | trust | 192 | 0.5052 | 0.4323 | 0.578 |
| valence | sharedreward | 192 | 0.8646 | 0.8079 | 0.9096 |
| valence | trust | 192 | 0.9531 | 0.9129 | 0.9783 |

Context averages outcomes; positive-context discrimination compares social wins to nonsocial wins; the sociality × valence interaction compares the positive-minus-negative difference between categories. Context-to-interaction prediction is not a success criterion. Weak interaction decoding does not show that social/friend outcomes lack reward value; it indicates limited decodable differential outcome modulation under these tasks/models. Valence is a reused v4 comparator averaging C/F/S equally, whereas Figure A illustrates an equal-category factorial contrast.

## Does Doors share the organization or remain a boundary?

| train_domain | test_domain | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharedreward_positive | socialdoors_positive | 178 | 0.4494 | 0.3749 | 0.5256 | -0.6337 | -1.463 | 0.181 |
| sharedreward_positive | socialdoors_negative | 178 | 0.5056 | 0.4298 | 0.5812 | 0.1663 | -0.6115 | 0.9243 |
| sharedreward_negative | socialdoors_positive | 178 | 0.4157 | 0.3425 | 0.4918 | -1.23 | -2.002 | -0.4752 |
| sharedreward_negative | socialdoors_negative | 178 | 0.4551 | 0.3804 | 0.5312 | -0.7683 | -1.549 | -0.007992 |
| trust_positive | socialdoors_positive | 178 | 0.6854 | 0.6117 | 0.7528 | 0.8948 | 0.6003 | 1.195 |
| trust_positive | socialdoors_negative | 178 | 0.7247 | 0.6529 | 0.7889 | 1.128 | 0.8139 | 1.44 |
| trust_negative | socialdoors_positive | 178 | 0.6854 | 0.6117 | 0.7528 | 1.012 | 0.699 | 1.325 |
| trust_negative | socialdoors_negative | 178 | 0.7191 | 0.647 | 0.7838 | 1.255 | 0.9208 | 1.579 |
| socialdoors_positive | socialdoors_positive | 178 | 0.9382 | 0.8921 | 0.9688 | 1.225 | 1.097 | 1.358 |
| socialdoors_positive | socialdoors_negative | 178 | 0.927 | 0.8783 | 0.9605 | 1.225 | 1.104 | 1.352 |
| socialdoors_negative | socialdoors_positive | 178 | 0.927 | 0.8783 | 0.9605 | 1.243 | 1.117 | 1.371 |
| socialdoors_negative | socialdoors_negative | 178 | 0.9326 | 0.8852 | 0.9647 | 1.245 | 1.123 | 1.369 |

Read the two within-Doors cross-valence cells separately from SR/Trust-to-Doors transfer. This secondary N=178 matrix does not redefine the primary N=192 cohort, construct or feature space. The full six-domain matrix is supplied in the supplement.

## Prespecified participant-label permutation diagnostics

| family | train_domain | test_domain | n | permutations | observed_accuracy | null_mean | null_sd | null_q025 | null_q975 | empirical_p | chance_centering_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| social_context | sharedreward_positive | trust_negative | 192 | 500 | 0.8385 | 0.5003 | 0.03711 | 0.4271 | 0.5757 | 0.001996 | False |
| social_context | sharedreward_negative | trust_positive | 192 | 500 | 0.849 | 0.4986 | 0.03681 | 0.4348 | 0.5677 | 0.001996 | False |
| social_context | trust_positive | sharedreward_negative | 192 | 500 | 0.7448 | 0.4993 | 0.03759 | 0.4167 | 0.5704 | 0.001996 | False |
| social_context | trust_negative | sharedreward_positive | 192 | 500 | 0.7917 | 0.4984 | 0.0353 | 0.4271 | 0.5625 | 0.001996 | False |
| social_context | pooled_positive | combined | 192 | 500 | 0.8906 | 0.5 | 0.026 | 0.4518 | 0.5495 | 0.001996 | False |
| social_context | pooled_negative | combined | 192 | 500 | 0.8776 | 0.4984 | 0.02596 | 0.4518 | 0.5521 | 0.001996 | False |
| friend_stranger_context | sharedreward_positive | trust_negative | 192 | 500 | 0.849 | 0.498 | 0.03591 | 0.4323 | 0.5677 | 0.001996 | False |
| friend_stranger_context | sharedreward_negative | trust_positive | 192 | 500 | 0.7656 | 0.5002 | 0.03478 | 0.4323 | 0.5729 | 0.001996 | False |
| friend_stranger_context | trust_positive | sharedreward_negative | 192 | 500 | 0.6719 | 0.5004 | 0.03469 | 0.4375 | 0.5677 | 0.001996 | False |
| friend_stranger_context | trust_negative | sharedreward_positive | 192 | 500 | 0.6979 | 0.4979 | 0.03594 | 0.4271 | 0.5677 | 0.001996 | False |
| friend_stranger_context | collapsed_common | sharedreward | 192 | 500 | 0.8542 | 0.4996 | 0.04238 | 0.4191 | 0.5861 | 0.001996 | False |
| friend_stranger_context | collapsed_common | trust | 192 | 500 | 0.875 | 0.502 | 0.03912 | 0.4219 | 0.5729 | 0.001996 | False |

One participant flip is shared across all tasks/valences and applied to training and test labels; five folds are refit. The 12 prespecified endpoints use 500 permutations each (shared fits where endpoints have identical training data). Empirical p=(1+null≥observed)/(1+B), with resolution 1/501. These are development diagnostics without project-wide multiplicity control. Combined-task accuracy averages correctness within each participant; it is not the fraction with a positive task-averaged margin.

All null means are within the descriptive [.45,.55] check.

## Alternate QC and remaining interpretation limits

No new cross-valence models were run under the alternate QC policy. Earlier characterization robustness cannot establish robustness of these new endpoints. Shared Reward is full-trial; Trust and Doors are outcome-phase. Perceptual/identity/familiarity and phase differences remain alternatives. Existing weak run-level reliability still limits individual-difference or trait claims.

## Which exact signatures could be frozen?

See [FREEZE_CANDIDATES.md](FREEZE_CANDIDATES.md) for the original pooled SR–Trust social-context model and the new pooled valence-collapsed friend–stranger model, exact inputs, coefficients, intercepts, masks and SHA256 fingerprints. Review the results before deciding endpoints; no automatic accuracy cutoff selects a winner. Neither candidate has been validated.

See [VISUAL_SENSITIVITY_PROPOSAL.md](VISUAL_SENSITIVITY_PROPOSAL.md) for a local independent atlas inventory and proposed exclusion. **No visual exclusion analysis was executed.**

- [FigureA_factorial PNG](../../../results/revised/cross_valence/figures/FigureA_factorial.png) · [PDF](../../../results/revised/cross_valence/figures/FigureA_factorial.pdf)
- [FigureB_social_context PNG](../../../results/revised/cross_valence/figures/FigureB_social_context.png) · [PDF](../../../results/revised/cross_valence/figures/FigureB_social_context.pdf)
- [FigureC_friend_stranger PNG](../../../results/revised/cross_valence/figures/FigureC_friend_stranger.png) · [PDF](../../../results/revised/cross_valence/figures/FigureC_friend_stranger.pdf)
- [FigureD_pooled PNG](../../../results/revised/cross_valence/figures/FigureD_pooled.png) · [PDF](../../../results/revised/cross_valence/figures/FigureD_pooled.pdf)
- [FigureE_patterns PNG](../../../results/revised/cross_valence/figures/FigureE_patterns.png) · [PDF](../../../results/revised/cross_valence/figures/FigureE_patterns.pdf)
- [supplement_Doors_matrix PNG](../../../results/revised/cross_valence/figures/supplement_Doors_matrix.png) · [PDF](../../../results/revised/cross_valence/figures/supplement_Doors_matrix.pdf)
- [supplement_spatial_similarity PNG](../../../results/revised/cross_valence/figures/supplement_spatial_similarity.png) · [PDF](../../../results/revised/cross_valence/figures/supplement_spatial_similarity.pdf)

PNG figures use 300 dpi; charts retain vector text in PDF/SVG. Original v4 and characterization artifacts are checked unchanged after rendering. **holdout_scored=False; primary validation N=50 untouched.**
