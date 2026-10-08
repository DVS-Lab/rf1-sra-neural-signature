# Final development specificity package

Run only this package on Linux2 after updating `main`. It never calls the UGR/betrayal runner or writes frozen candidates. Local implementation tests use synthetic inputs; scientific output requires the private Linux2 development manifests, source maps, and original cross-valence cache.

```bash
cd /ZPOOL/data/projects/rf1-sra-neural-signature
git pull --ff-only
bash code/run_specificity.sh --dry-run
bash code/run_specificity.sh --workers 96
```

If this clone predates the already-announced correspondence history cleanup, first reconcile it to the rewritten `origin/main`; do not merge the old history back. Preserve any uncommitted work before doing so. This package does not itself rewrite Git history.

The dry run checks original membership/folds/QC, frozen files, upstream phase provenance and hashes the allowed development source images; it does not load voxel arrays. Real runs retain exactly the original 178-person three-paradigm sample. Missing maps or changed QC stop the run rather than silently reduce N, redraw folds, relax exclusions or touch validation. Reads of protected membership metadata only enforce exclusion; no protected image or score is read. Source maps and cache arrays remain in ignored `work/`.

Two analyses are fixed before execution:

1. Repeat the six-domain social-context matrix: positive and negative outcomes in Shared Reward, Trust, and Social Doors versus monetary Doors. SR uses upstream phase-resolved COPEs 1–6 (`C_pun`, `C_rew`, `F_pun`, `F_rew`, `S_pun`, `S_rew`); positive human/computer maps are `(COPE4+COPE6)/2` versus `COPE2`, negative maps `(COPE3+COPE5)/2` versus `COPE1`. Trust already models outcomes: positive human/computer `(COPE7+COPE9)/2` versus `COPE5`; negative `(COPE6+COPE8)/2` versus `COPE4`. Doors uses win COPE1 and loss COPE2 in each paradigm. Reproduce previous full-trial SR OOF predictions exactly within stated numerical tolerance. Retain all four within-Doors same/cross-valence controls.
2. Quantify threshold-free spatial overlap of the two unchanged N=192 frozen candidate weight/forward/mean-difference maps with independently defined cortical DMN. Evaluate fixed DMN-only, DMN-excluded and generic task-response projections against outcome whole-mask models, all on matched N=178. Both human–computer and friend–stranger outcome transfers are included. Friend–stranger uses F versus S at each valence; its full-trial comparison is a matched N=178 refit, not a comparison to the old N=192 accuracy. Doors has no friend–stranger contrast.

Reference: [Yeo et al. (2011), official atlas documentation](https://freesurfer.net/fswiki/CorticalParcellation_Yeo2011). The bundled seven-network liberal-mask atlas in FSL nonlinear MNI152 coordinates uses fixed label 7, Default. Original singleton fourth dimension is squeezed; nearest-neighbor resampling preserves integer network labels. The original download URL and SHA256 are pinned in `config/specificity.json`; the distribution README is retained. This cortical resting-state reference is not synonymous with all task-negative regions. The complement retains every original-mask voxel outside label 7, including unlabelled and noncortical voxels. No atlas variants, threshold search, or performance-driven masks are available.

Generic response template = equally weighted training-person, training-task condition mean (all six partner×valence maps in SR/Trust; four social/monetary×valence maps in Doors), independently spatially centered and unit normalized. It is calculated without class labels and without test participants. Fixed C=1 1-D LinearSVC learns only a score coefficient/intercept on training projections. Test-task maps never contribute to the training template. This internal task-response direction is not an independent external template. All other classifiers retain the original LinearSVC parameters and per-map spatial centering, re-applied inside restricted masks. No full-development replacement candidates are produced.

Accuracy retains the previous paired-ordering definition: the fraction of participant pairs with score(human)>score(computer), or score(friend)>score(stranger), with ties counted as incorrect.

Nine central endpoints: six directed social-context transfers, the within-Doors control, and two directed friend–stranger transfers. Average the four train/test-valence cell correctness values *within participant*. All five model variants share N, participant order, folds and permutation flips. Full 52-cell results per variant remain available to reveal valence asymmetry; no cell is chosen based on its accuracy.

Inference: 10,000 paired participant bootstraps of fixed OOF correctness for descriptive accuracy/change intervals; cellwise exact binomial OOF intervals. These intervals condition on the fitted predictions and do not include full model-retraining variability. Perform 500 synchronized subject-label permutations per variant, refitting all five folds and test-label orientation. Report raw above-chance and two-sided p values, synchronized max-|accuracy−0.5| correction across all 45 variant/endpoint combinations, null mean/SD/quantiles/unique values, and Monte Carlo tail intervals. The max-statistic result is a complete-null familywise diagnostic, not a claim of strong control under every mixture of alternatives. Permutations test discrimination, not between-variant accuracy differences. A null-center flag is diagnostic only and never changes the model or mask.

The runner requests 96 processes, caps by detected CPUs/available RAM, and limits BLAS to one thread per process. Each task is an indexed permutation/variant, with shared read-only memmaps and no full-cohort masked copies per worker. Completed observed predictions and indexed permutations resume from fingerprinted private checkpoints. All inherited code and public artifacts are pinned; a new package fingerprint binds private membership, folds, source hashes, software, QC and implementation. No automatic checkpoint invalidation or deletion occurs.

Outputs:

- `reports/revised/specificity/SPECIFICITY.md`: one concise report, results, limits and remaining preregistration decisions.
- `results/revised/specificity/figures/specificity_comparison.png` and `.pdf`: one two-panel figure.
- `results/revised/specificity/aggregate/`: central results, full matrices, paired differences, overlap, null diagnostics and old-prediction reconstruction.
- `provenance/revised/specificity/run_status.json`: completion and original-output invariance.

Stop after these outputs. Preregistration still requires decisions on the primary construct/candidate and full-trial versus outcome-only status, external-validation mapping/QC, confirmatory endpoints/multiplicity/success criteria, and timing/procedure for protected N=50 scoring. Survival outside DMN does not establish an abstract social representation. Correspondence belongs outside this repository.

## Generic-template numerical repair

The first production run reproduced the old predictions and completed the four voxel-model variants, then stopped because the dual optimizer did not converge on the generic one-feature projections. The generic baseline now uses `LinearSVC(dual=False)` consistently for observed and permuted fits. This solves the same L2 squared-hinge objective, with C=1, tolerance=0.0001, max_iter=100000 and unchanged intercept penalty; no feature rescaling, accuracy tuning or warning suppression is introduced. The voxel models retain their original dual solver. See [the official solver guidance](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

After `git pull --ff-only`, use the same launcher. A narrow automatic restart accepts only the exact original implementation hashes and identical data, software, QC, masks and folds; it checks the feature cache and all four completed prediction arrays. The original identity/markers stay intact. A separate `solver_restart.json` records authorization to reuse those products with their original fingerprint. Generic fits and every permutation use the repaired execution fingerprint. Existing old generic/null products or any unrelated input/code drift cause a stop. Interrupted repairs can resume; do not delete or edit checkpoints.
