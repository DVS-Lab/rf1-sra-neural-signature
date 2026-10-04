# Candidate signatures for prospective validation review

These exact development artifacts are candidates, not validated signatures or automatic winners. Review the directional performance, uncertainty, permutation diagnostics, construct boundaries and intended endpoints before preregistration. No single accuracy cutoff selects a model.

## Social context

Reused original v4 common model. SR: mean[(F_rew+F_pun)/2, (S_rew+S_pun)/2] versus (C_rew+C_pun)/2; Trust: replace rew/pun with rec/def. Positive class = human context; negative class = computer context. SR COPEs: C=2/1, F=4/3, S=6/5; Trust: C=5/4, F=7/6, S=9/8 (positive/negative).

Training: all 192 primary development participants; both tasks; no protected validation participant. Shared Reward full-trial retained-run strategy and Trust outcome inputs are unchanged.

- Weight file: `results/revised/cross_valence/maps/partner_pair_social_context_collapsed_common_DEV_weights.nii.gz`
- Weight SHA256: `27e20e778fd84355097fc3e020e96f93b850c5807b5b9125da99d719b9058d99`
- Mask: `results/revised/cross_valence/maps/partner_pair_analysis_mask.nii.gz`
- Mask SHA256: `744d21ac3dfa4f7b6fbd0003789b4b6c6ae2a8a6d7e240dbe420fef3d45f9462`
- Intercept: `-4.167728242588583e-07`
- Exact model manifest: `provenance/revised/cross_valence/candidate_social_context.json`
- Model manifest SHA256: `70e3bc7048eaa0eef7eb77ae2f0c5b857e08ee99032f1a7fa2afbdefe7a578b2`

Preprocessing: independent spatial mean centering inside the frozen partner-pair mask; no PCA, feature selection or scaling. LinearSVC(C=1, dual=True, class_weight=None, max_iter=100000, tol=.0001, random_state=20260928). NIfTI coefficients are float32; the manifest preserves the fitted intercept.

## Friend–stranger context

New pooled valence-collapsed candidate. SR: (F_rew+F_pun)/2 versus (S_rew+S_pun)/2; Trust: (F_rec+F_def)/2 versus (S_rec+S_def)/2. Positive class = friend; negative class = stranger. SR COPEs: F=4/3, S=6/5; Trust: F=7/6, S=9/8 (positive/negative). Cross-valence support must be assessed from the full directional results, not assumed from the candidate file existing.

Training: all 192 primary development participants; both tasks; no protected validation participant. Shared Reward full-trial retained-run strategy and Trust outcome inputs are unchanged.

- Weight file: `results/revised/cross_valence/maps/partner_pair_friend_stranger_context_collapsed_common_DEV_weights.nii.gz`
- Weight SHA256: `9a8a0328504affd41b0bd3a507be853a3f8f17442ef7fdbec8f0ba3d3ec6bb93`
- Mask: `results/revised/cross_valence/maps/partner_pair_analysis_mask.nii.gz`
- Mask SHA256: `744d21ac3dfa4f7b6fbd0003789b4b6c6ae2a8a6d7e240dbe420fef3d45f9462`
- Intercept: `-1.151673290626014e-06`
- Exact model manifest: `provenance/revised/cross_valence/candidate_friend_stranger_context.json`
- Model manifest SHA256: `b3a9392883e3a8544163055d34527ea7256954965abc78263e724e8c407d12b6`

Preprocessing: independent spatial mean centering inside the frozen partner-pair mask; no PCA, feature selection or scaling. LinearSVC(C=1, dual=True, class_weight=None, max_iter=100000, tol=.0001, random_state=20260928). NIfTI coefficients are float32; the manifest preserves the fitted intercept.

## Comparators and boundaries

Outcome valence, sociality × valence interaction, friend–stranger × valence interaction, and Doors transfer remain explicit comparisons/boundaries. Their completed v4 results are not retuned.

Before scoring N=50: specify exact models, contrasts, transfer endpoints, success criteria and multiplicity; review whether a separately approved visual-control analysis is needed. Neither pure closeness, abstract social representation, trait-like social sensitivity, nor a social reward signature is established automatically.

holdout_scored=False.
