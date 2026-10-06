# Final development stress tests

Occipital exclusion is descriptive only: no model selection, no persistence success criterion, and no proof of abstract social coding. Phase-resolved SR outcome ↔ Trust outcome generalization is the primary stress test. It addresses a simple concurrent face/nonface explanation, while hemodynamic carryover, identity, familiarity, text, semantics and other perceptual information remain alternatives. UGR constant maps are visually confounded third-paradigm probes. Frozen DEV applications involve training-participant overlap; only participant-blocked CV endpoints test unseen development participants. The protected N=50 validation sample remains unscored.

Current-source/QC subsets of the frozen N=192 development pool; no roster reassignment. Unknown QC never passes. Computational minimum is ten people and all five original folds, not a power guarantee. Original 95% coverage mask includes development CV test participants without labels and is retained; it is not fold-specific.

## 1. What was on screen?

See [TASK_PHASE_AUDIT.md](TASK_PHASE_AUDIT.md) and Figure 1. SR outcome: generic card/outcome display without partner name/face; Trust outcome: partner text and valence symbols without a face; UGR: face/computer cues remain during offers. Presentation code verifies the checked version, not every historical deployment.

## 2. Can phase-resolved SR outcomes decode social context?

| model | test | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high | evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_social_context_1_whole_brain | sr_outcome_social_context | 192 | 0.8646 | 0.8079 | 0.9096 | 0.6275 | 0.5329 | 0.7259 | participant_blocked_cv |

This is within-domain participant-blocked decoding; the partner decision events are separately modeled.

## 3. Does outcome-only social context transfer between SR and Trust?

| model | test | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high | evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_social_context_1_whole_brain | trust_social_context | 192 | 0.8906 | 0.8377 | 0.931 | 0.3396 | 0.2983 | 0.3814 | participant_blocked_cv |
| phase_social_context_2_whole_brain | sr_outcome_social_context | 192 | 0.7865 | 0.7217 | 0.8422 | 1.608 | 1.288 | 1.927 | participant_blocked_cv |

These matched-sample participant-blocked tests lead the interpretation. Positive evidence would argue against a simple concurrent face/nonface explanation across both target epochs; it would not eliminate carryover, semantic or identity alternatives.

## 4. Does friend–stranger context show outcome-only transfer?

| model | test | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high | evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_friend_stranger_context_0_whole_brain | sr_full_friend_stranger_context | 192 | 0.8438 | 0.7845 | 0.892 | 0.8001 | 0.6853 | 0.9157 | participant_blocked_cv |
| phase_friend_stranger_context_0_whole_brain | sr_outcome_friend_stranger_context | 192 | 0.7656 | 0.6992 | 0.8236 | 3.451 | 2.742 | 4.135 | participant_blocked_cv |
| phase_friend_stranger_context_0_whole_brain | trust_friend_stranger_context | 192 | 0.8854 | 0.8317 | 0.9268 | 1.907 | 1.664 | 2.154 | participant_blocked_cv |
| phase_friend_stranger_context_1_whole_brain | sr_full_friend_stranger_context | 192 | 0.7812 | 0.716 | 0.8376 | 0.08135 | 0.06666 | 0.09634 | participant_blocked_cv |
| phase_friend_stranger_context_1_whole_brain | sr_outcome_friend_stranger_context | 192 | 0.7812 | 0.716 | 0.8376 | 0.5762 | 0.4739 | 0.6757 | participant_blocked_cv |
| phase_friend_stranger_context_1_whole_brain | trust_friend_stranger_context | 192 | 0.8542 | 0.7962 | 0.9009 | 0.2332 | 0.1981 | 0.2688 | participant_blocked_cv |
| phase_friend_stranger_context_2_whole_brain | sr_full_friend_stranger_context | 192 | 0.8177 | 0.7557 | 0.8696 | 0.2879 | 0.2363 | 0.3415 | participant_blocked_cv |
| phase_friend_stranger_context_2_whole_brain | sr_outcome_friend_stranger_context | 192 | 0.7344 | 0.666 | 0.7954 | 1.458 | 1.109 | 1.807 | participant_blocked_cv |
| phase_friend_stranger_context_2_whole_brain | trust_friend_stranger_context | 192 | 0.9323 | 0.887 | 0.9635 | 1.179 | 1.05 | 1.309 | participant_blocked_cv |
| phase_friend_stranger_context_pooled_whole_brain | sr_outcome_friend_stranger_context | 192 | 0.7812 | 0.716 | 0.8376 | 1.526 | 1.223 | 1.825 | participant_blocked_cv |
| phase_friend_stranger_context_pooled_whole_brain | trust_friend_stranger_context | 192 | 0.9375 | 0.8934 | 0.9673 | 1.054 | 0.9462 | 1.162 | participant_blocked_cv |

The full three-domain matrix includes the matched full-trial reference, outcome-only bidirectional transfer and separate pooled sensitivity models. None replaces the frozen candidate.

## 5. Predefined occipital exclusion

| model | test | n | original_accuracy | visual_excluded_accuracy | difference | retained_voxels | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| architecture_friend_stranger_context_visual_excluded | sr_full_friend_stranger_context | 192 | 0.8542 | 0.8542 | 0 | 74373 | descriptive_only_not_selection_or_success_gate |
| architecture_friend_stranger_context_visual_excluded | trust_friend_stranger_context | 192 | 0.875 | 0.875 | 0 | 74373 | descriptive_only_not_selection_or_success_gate |
| architecture_social_context_visual_excluded | sr_full_social_context | 192 | 0.9635 | 0.9479 | -0.01562 | 74373 | descriptive_only_not_selection_or_success_gate |
| architecture_social_context_visual_excluded | trust_social_context | 192 | 0.9531 | 0.9323 | -0.02083 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_friend_stranger_context_0_visual_excluded | sr_full_friend_stranger_context | 192 | 0.8438 | 0.8385 | -0.005208 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_friend_stranger_context_0_visual_excluded | sr_outcome_friend_stranger_context | 192 | 0.7656 | 0.7708 | 0.005208 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_friend_stranger_context_0_visual_excluded | trust_friend_stranger_context | 192 | 0.8854 | 0.8958 | 0.01042 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_friend_stranger_context_1_visual_excluded | sr_full_friend_stranger_context | 192 | 0.7812 | 0.8073 | 0.02604 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_friend_stranger_context_1_visual_excluded | sr_outcome_friend_stranger_context | 192 | 0.7812 | 0.7708 | -0.01042 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_friend_stranger_context_1_visual_excluded | trust_friend_stranger_context | 192 | 0.8542 | 0.8646 | 0.01042 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_friend_stranger_context_2_visual_excluded | sr_full_friend_stranger_context | 192 | 0.8177 | 0.7969 | -0.02083 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_friend_stranger_context_2_visual_excluded | sr_outcome_friend_stranger_context | 192 | 0.7344 | 0.7344 | 0 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_friend_stranger_context_2_visual_excluded | trust_friend_stranger_context | 192 | 0.9323 | 0.9271 | -0.005208 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_friend_stranger_context_pooled_visual_excluded | sr_outcome_friend_stranger_context | 192 | 0.7812 | 0.7969 | 0.01562 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_friend_stranger_context_pooled_visual_excluded | trust_friend_stranger_context | 192 | 0.9375 | 0.9323 | -0.005208 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_social_context_0_visual_excluded | sr_full_social_context | 192 | 0.9583 | 0.9479 | -0.01042 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_social_context_0_visual_excluded | sr_outcome_social_context | 192 | 0.6823 | 0.6875 | 0.005208 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_social_context_0_visual_excluded | trust_social_context | 192 | 0.9375 | 0.9219 | -0.01562 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_social_context_1_visual_excluded | sr_full_social_context | 192 | 0.75 | 0.7708 | 0.02083 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_social_context_1_visual_excluded | sr_outcome_social_context | 192 | 0.8646 | 0.8177 | -0.04688 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_social_context_1_visual_excluded | trust_social_context | 192 | 0.8906 | 0.901 | 0.01042 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_social_context_2_visual_excluded | sr_full_social_context | 192 | 0.8854 | 0.8646 | -0.02083 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_social_context_2_visual_excluded | sr_outcome_social_context | 192 | 0.7865 | 0.776 | -0.01042 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_social_context_2_visual_excluded | trust_social_context | 192 | 0.9427 | 0.9479 | 0.005208 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_social_context_pooled_visual_excluded | sr_outcome_social_context | 192 | 0.8021 | 0.8021 | 0 | 74373 | descriptive_only_not_selection_or_success_gate |
| phase_social_context_pooled_visual_excluded | trust_social_context | 192 | 0.9583 | 0.9479 | -0.01042 | 74373 | descriptive_only_not_selection_or_success_gate |
| sr_outcome_to_ugr_visual_excluded | ugr_context | 176 | 0.4545 | 0.6875 | 0.233 | 74373 | descriptive_only_not_selection_or_success_gate |
| sr_outcome_to_ugr_visual_excluded | ugr_high | 176 | 0.5284 | 0.6364 | 0.108 | 74373 | descriptive_only_not_selection_or_success_gate |
| sr_outcome_to_ugr_visual_excluded | ugr_low | 176 | 0.4773 | 0.642 | 0.1648 | 74373 | descriptive_only_not_selection_or_success_gate |
| trust_to_ugr_visual_excluded | ugr_context | 176 | 0.8864 | 0.8864 | 0 | 74373 | descriptive_only_not_selection_or_success_gate |
| trust_to_ugr_visual_excluded | ugr_high | 176 | 0.7841 | 0.8068 | 0.02273 | 74373 | descriptive_only_not_selection_or_success_gate |
| trust_to_ugr_visual_excluded | ugr_low | 176 | 0.8636 | 0.8466 | -0.01705 | 74373 | descriptive_only_not_selection_or_success_gate |

Descriptive sensitivity only. Neither persistence nor failure determines candidate selection or success.

## 6. Distinguishable context dimensions?

| model | test | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high | evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| friend_stranger_context | sr_full_friend_vs_stranger | 192 | 0.8542 | 0.7962 | 0.9009 | 0.7553 | 0.6473 | 0.8635 | participant_blocked_cv |
| friend_stranger_context | sr_full_human_vs_computer | 192 | 0.8594 | 0.802 | 0.9052 | 0.592 | 0.5134 | 0.6708 | participant_blocked_cv |
| friend_stranger_context | trust_friend_vs_stranger | 192 | 0.875 | 0.8197 | 0.9182 | 2.028 | 1.793 | 2.269 | participant_blocked_cv |
| friend_stranger_context | trust_human_vs_computer | 192 | 0.9271 | 0.8807 | 0.9596 | 1.767 | 1.565 | 1.971 | participant_blocked_cv |
| social_context | sr_full_friend_vs_stranger | 192 | 0.7969 | 0.733 | 0.8514 | 0.8028 | 0.6767 | 0.9327 | participant_blocked_cv |
| social_context | sr_full_human_vs_computer | 192 | 0.9635 | 0.9263 | 0.9852 | 1.13 | 1.03 | 1.234 | participant_blocked_cv |
| social_context | trust_friend_vs_stranger | 192 | 0.875 | 0.8197 | 0.9182 | 2.07 | 1.806 | 2.328 | participant_blocked_cv |
| social_context | trust_human_vs_computer | 192 | 0.9531 | 0.9129 | 0.9783 | 2.258 | 2.042 | 2.47 | participant_blocked_cv |

Figure 4 shows paired C/S/F expression. Above-chance cross-application is overlap, not independence. A nonsignificant F–S result does not establish F≈S; no equivalence threshold was specified. Compare profiles and both native/cross-application contrasts; margins across separately trained models are not on a common scale.

## 7. Frozen social context → UGR

| model | test | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high | evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_DEV_friend_stranger_context | sr_outcome_friend_stranger_context | 192 | 1 | 0.981 | 1 | 9.044 | 8.578 | 9.522 | development_training_participant_overlap |
| frozen_DEV_social_context | sr_outcome_social_context | 192 | 0.9635 | 0.9263 | 0.9852 | 6.53 | 6.018 | 7.037 | development_training_participant_overlap |
| frozen_DEV_social_context | ugr_context | 176 | 0.9943 | 0.9688 | 0.9999 | 1.9 | 1.747 | 2.06 | development_training_participant_overlap |
| frozen_DEV_social_context | ugr_high | 176 | 0.9602 | 0.9198 | 0.9839 | 1.904 | 1.728 | 2.087 | development_training_participant_overlap |
| frozen_DEV_social_context | ugr_low | 176 | 0.9886 | 0.9596 | 0.9986 | 1.896 | 1.735 | 2.066 | development_training_participant_overlap |

These are fixed-weight development probes with training-participant overlap, including phase-resolved SR probes. UGR is visually confounded; success alone cannot adjudicate perception.

## 8. Outcome-trained context → UGR

| model | test | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high | evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sr_outcome_to_ugr_visual_excluded | ugr_context | 176 | 0.6875 | 0.6134 | 0.7551 | 0.04852 | 0.03502 | 0.06211 | participant_blocked_cv |
| sr_outcome_to_ugr_visual_excluded | ugr_high | 176 | 0.6364 | 0.5606 | 0.7074 | 0.05058 | 0.03291 | 0.0681 | participant_blocked_cv |
| sr_outcome_to_ugr_visual_excluded | ugr_low | 176 | 0.642 | 0.5665 | 0.7128 | 0.04645 | 0.02874 | 0.06432 | participant_blocked_cv |
| sr_outcome_to_ugr_whole_brain | ugr_context | 176 | 0.4545 | 0.3795 | 0.5312 | 0.007674 | -0.005864 | 0.02168 | participant_blocked_cv |
| sr_outcome_to_ugr_whole_brain | ugr_high | 176 | 0.5284 | 0.4519 | 0.604 | 0.009264 | -0.008223 | 0.02693 | participant_blocked_cv |
| sr_outcome_to_ugr_whole_brain | ugr_low | 176 | 0.4773 | 0.4016 | 0.5537 | 0.006084 | -0.01195 | 0.02462 | participant_blocked_cv |
| trust_to_ugr_visual_excluded | ugr_context | 176 | 0.8864 | 0.83 | 0.9292 | 0.4028 | 0.3479 | 0.4567 | participant_blocked_cv |
| trust_to_ugr_visual_excluded | ugr_high | 176 | 0.8068 | 0.7407 | 0.8624 | 0.3969 | 0.3281 | 0.4657 | participant_blocked_cv |
| trust_to_ugr_visual_excluded | ugr_low | 176 | 0.8466 | 0.7847 | 0.8964 | 0.4087 | 0.3478 | 0.47 | participant_blocked_cv |
| trust_to_ugr_whole_brain | ugr_context | 176 | 0.8864 | 0.83 | 0.9292 | 0.3885 | 0.3332 | 0.4427 | participant_blocked_cv |
| trust_to_ugr_whole_brain | ugr_high | 176 | 0.7841 | 0.7159 | 0.8424 | 0.3818 | 0.3122 | 0.4507 | participant_blocked_cv |
| trust_to_ugr_whole_brain | ugr_low | 176 | 0.8636 | 0.8039 | 0.9106 | 0.3951 | 0.3335 | 0.4572 | participant_blocked_cv |

Participant blocking is enforced. Training excludes concurrently displayed faces, but the UGR target remains visually confounded.

## 9. Trust social defection/reciprocation → UGR unfair/fair?

| model | test | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high | evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trust_to_ugr_norm_whole_brain | ugr_computer_norm | 176 | 0.6932 | 0.6194 | 0.7604 | 0.1932 | 0.1319 | 0.2547 | participant_blocked_cv |
| trust_to_ugr_norm_whole_brain | ugr_norm | 176 | 0.7386 | 0.6672 | 0.8019 | 0.2593 | 0.1971 | 0.3219 | participant_blocked_cv |

Categorical UGR status: complete. 
The social model averages friend and stranger outcomes. Its UGR nonsocial application is a separate specificity probe; fair/unfair is defined by nominal offer categories, never the response.

## 10. Does reverse UGR → Trust transfer work?

| model | test | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high | evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ugr_to_trust_norm_whole_brain | trust_norm | 176 | 0.875 | 0.8169 | 0.92 | 1.072 | 0.9242 | 1.223 | participant_blocked_cv |

Positive orientation remains unfair/defection; all test participants are absent from training.

## 11. Generic valence and nonsocial comparisons

| model | test | n | accuracy | ci_low | ci_high | mean_margin | margin_ci_low | margin_ci_high | evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| computer_norm_control_whole_brain | ugr_computer_norm | 176 | 0.6705 | 0.5957 | 0.7393 | 0.09911 | 0.06131 | 0.1375 | participant_blocked_cv |
| existing_valence_OOF_reversed | trust_norm | 176 | 0.9432 | 0.898 | 0.9724 | 2.152 | 1.937 | 2.372 | participant_blocked_existing_OOF |
| existing_valence_OOF_reversed | ugr_norm | 176 | 0.6534 | 0.5781 | 0.7234 | 0.3139 | 0.1849 | 0.4426 | participant_blocked_existing_OOF |

Generic valence uses existing OOF models with score sign reversed so positive predicts violation. It need not fail. Margins across models lack a common scale; compare forced-choice accuracy descriptively on matched participants. Fairness amount, valence and choice can contribute; regional overlap or lack of regional correlations does not settle the distributed question.

## 12. Collaborator summary and eventual preregistration

The two frozen candidates remain unchanged. These are final development diagnostics; none is an automatic validation gate. Review directional phase-transfer estimates and intervals, construct cross-application, UGR boundaries and the separately labeled exploratory norm family. Preserve the untouched N=50 and preregister endpoints, uncertainty/success rules and multiplicity before validation.

## Participant permutations

| model | test | n | permutations | observed_accuracy | null_mean | null_sd | empirical_p | chance_centering_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_social_context_1_whole_brain | trust_social_context | 192 | 500 | 0.8906 | 0.4981 | 0.03346 | 0.001996 | False |
| phase_social_context_2_whole_brain | sr_outcome_social_context | 192 | 500 | 0.7865 | 0.5008 | 0.03333 | 0.001996 | False |
| phase_friend_stranger_context_1_whole_brain | trust_friend_stranger_context | 192 | 500 | 0.8542 | 0.5017 | 0.03721 | 0.001996 | False |
| phase_friend_stranger_context_2_whole_brain | sr_outcome_friend_stranger_context | 192 | 500 | 0.7344 | 0.5017 | 0.03915 | 0.001996 | False |
| trust_to_ugr_whole_brain | ugr_context | 176 | 500 | 0.8864 | 0.5002 | 0.03883 | 0.001996 | False |
| trust_to_ugr_norm_whole_brain | ugr_norm | 176 | 500 | 0.7386 | 0.5024 | 0.03779 | 0.001996 | False |
| ugr_to_trust_norm_whole_brain | trust_norm | 176 | 500 | 0.875 | 0.5004 | 0.0381 | 0.001996 | False |

500 flips per primary new endpoint; a participant shares one flip across tasks in a family, including test labels; all five folds refit. Nominal development diagnostics, not project-wide multiplicity control.

## Spatial comparisons

| comparison | kind | spatial_r | n_voxels |
| --- | --- | --- | --- |
| frozen_context_vs_friend_stranger | weight | 0.1998 | 84133 |
| frozen_context_vs_friend_stranger | haufe | 0.886 | 84133 |
| frozen_context_vs_friend_stranger | mean_difference | 0.8739 | 84133 |
| phase_social_context_0_whole_vs_visual_excluded | weight_on_retained_voxels | 0.992 | 74373 |
| phase_social_context_1_whole_vs_visual_excluded | weight_on_retained_voxels | 0.9898 | 74373 |
| phase_social_context_2_whole_vs_visual_excluded | weight_on_retained_voxels | 0.9848 | 74373 |
| phase_social_context_pooled_whole_vs_visual_excluded | weight_on_retained_voxels | 0.9848 | 74373 |
| architecture_social_context_whole_vs_visual_excluded | weight_on_retained_voxels | 0.9867 | 74373 |
| phase_friend_stranger_context_0_whole_vs_visual_excluded | weight_on_retained_voxels | 0.9922 | 74373 |
| phase_friend_stranger_context_1_whole_vs_visual_excluded | weight_on_retained_voxels | 0.9936 | 74373 |
| phase_friend_stranger_context_2_whole_vs_visual_excluded | weight_on_retained_voxels | 0.9841 | 74373 |
| phase_friend_stranger_context_pooled_whole_vs_visual_excluded | weight_on_retained_voxels | 0.9838 | 74373 |
| architecture_friend_stranger_context_whole_vs_visual_excluded | weight_on_retained_voxels | 0.9884 | 74373 |
| trust_to_ugr_whole_vs_visual_excluded | weight_on_retained_voxels | 0.9855 | 74373 |
| sr_outcome_to_ugr_whole_vs_visual_excluded | weight_on_retained_voxels | 0.9905 | 74373 |

Raw weights are not regional importance; Haufe and mean patterns are descriptive.

## Unavailable analyses

None.

No unavailable cell is encoded as zero or chance.

holdout_scored = False
