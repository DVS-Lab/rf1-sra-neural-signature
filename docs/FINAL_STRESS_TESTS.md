# Final development stress tests

This package is separate from v4, characterization and cross-valence. The primary test is phase-resolved Shared Reward outcome ↔ Trust outcome generalization. Occipital exclusion is descriptive only: it never selects a candidate, is never a required success criterion, and cannot establish abstract social coding. UGR fairness/norm violation is a separate exploratory family.

## Linux2 launch

From `/ZPOOL/data/projects/rf1-sra-neural-signature` on main, with the existing Python/FSL environment and original private manifests/checkpoints restored:

```bash
git pull --ff-only
bash code/run_final_stress_tests.sh --dry-run
bash code/run_final_stress_tests.sh --fairness-preview
```

The preview selects the first sorted **eligible development** UGR run, renders its actual FEAT design, checks trial counts, rank, correlations, VIF and efficiency, and writes `reports/revised/final_stress_tests/FAIRNESS_DESIGN_QC.md` plus `results/revised/final_stress_tests/figures/fairness_design_preview.png`. Inspect that image/report before the full launch:

```bash
bash code/run_final_stress_tests.sh --workers 96
```

The full launcher checks every categorical design before fitting any categorical L1. A failed scientific design gate explicitly stops the whole exploratory categorical branch; it never changes fairness cutoffs, drops individual difficult runs to rescue results, or replaces the fairness pmod. Independent context analyses still report. A missing FSL installation or mechanical execution error fails the run rather than being mislabeled a scientific null. The 50 protected validation participants are barred from inventory image inspection, BIDS-event access, feature loading, fitting, scoring and plots. No validation override exists.

Restart with the same command. Feature caches, CV coefficients and indexed permutations are fingerprinted; completed jobs are reused. FEAT completion is checked explicitly because FSL can submit jobs internally. An existing incomplete categorical FEAT directory stops rather than being deleted or silently overwritten; inspect the private log/output before repair. Changed code, membership, QC or inputs requires review and a fresh explicitly managed checkpoint, never automatic mixing of fitted results.

For a design-only preview, rerunning `--fairness-preview` after an update can archive the previous preview under private `work/revised/final_stress_tests/archived_previews/` and render it afresh. This requires only recognized preview artifacts: any features, fitted models, permutations, FEAT directory or unknown product blocks automatic archival. No files are deleted. This handles the initial preview's `dict() got multiple values for keyword argument 'failures'` reporting bug: pull the fix and rerun the preview without manually deleting checkpoints. Both passing and failed design-QC metrics now write correctly; a failed scientific design gate still stops the categorical branch.

```bash
bash code/run_final_stress_tests.sh --plots-only
```

Plots-only requires completed, hash-verified computations. It does not refit classifiers or run FEAT. Do not edit frozen v4/cross-valence artifacts to satisfy a failed check.

## Samples, maps and evidence types

Only the frozen N=192 primary development pool can contribute; original folds and all original holdout assignments remain fixed. Current exclusions and canonical `tsnr_outlier`, `brain_coverage_outlier`, `fd_mean_outlier` flags determine downstream availability. No new IQR fences are estimated. Unknown QC fails. Baseline SR uses its previously retained aging runs; Trust and UGR L2 use runs 1+2. Phase-resolved SR uses the completed October 5 RF1 activation audit: 655 runs / 346 subject-sessions, with 309 fixed-effects L2 outputs and 37 explicitly recorded L1 passthroughs. The adapter pins the verified manifest, summary and provenance, validates contributing runs, and checks QC for exactly those retained runs. It does not invent a single-run fallback from whichever directory happens to exist. Empty neutral/miss EVs are allowed only with canonical absence and shape 10; all six target outcome contrast vectors remain exact checks. Counts and missing reasons are printed **before fitting** and stored publicly in aggregate; membership stays private.

The launcher also prints directory-existence counts and writes `results/revised/final_stress_tests/aggregate/phase_directory_inventory.tsv`. It checks only frozen development participants, lists the expected phase L1/L2 paths and matching alternative activation directory names in the standard or `rf1/` layout, and never chooses alternative maps automatically. Existence is not proof of completed or valid maps. The full launch stops before feature loading if the primary phase sample lacks ten eligible development people or any frozen fold; the independent UGR design preview can still run. Aggregate retained-run/strategy counts are written to `results/revised/final_stress_tests/aggregate/phase_retained_runs.tsv`. The technical N=346 is not the classifier sample size: the frozen N=192 development scope and existing classifier QC still apply.

Phase matrices compare SR full-trial, SR outcome-only and Trust outcome on the identical intersection. Outcome-only SR+Trust pooled sensitivity models are separate artifacts. Original-architecture whole-brain and occipital-excluded models use a matched current-QC baseline subset. UGR transfers use source/target intersections, with the same person excluded across all tasks at test time. At least ten matched development people and all five original folds are required computationally; that is not a power guarantee.

Fixed final-DEV signatures applied to phase-resolved SR or UGR retain their exact weights. Those applications involve training-participant overlap and are **descriptive new-map probes**, not unseen-person validation. Their labels, tables and figures identify this explicitly. Generic valence uses original fold-specific coefficients, sign-reversed so positive corresponds to defection/unfairness, and is evaluated on the same norm-transfer people. Raw margins from different models do not have a shared scale.

The original analysis mask remains fixed. The exact eight Harvard-Oxford parcels and atlas/XML SHA256 values come from the approved cross-valence proposal; 9,760 of 84,133 primary-mask voxels are removed and 74,373 retained. Each map is recentered within the retained mask. Report original/excluded accuracy, difference, retained count and weight correlation on retained voxels; no success threshold is applied.

## Categorical UGR model

All outputs are under private `work/revised/final_stress_tests/fairness/`, named `model-signature-fairness`, and never inside upstream derivatives. Canonical BIDS only; no raw PsychoPy logs. The user approved nominal percentages with rounded dollars:

| Endowment | Unfair: nominal 5% / 10% | Fair: nominal 25% / 50% |
|---|---|---|
| $16 | $1 / $2 | $4 / $8 |
| $32 | $2 / $3 | $8 / $16 |

No unknown offer is rounded or reassigned after reading results. Every valid offer must match exactly one allowed amount. Four primary categorical regressors preserve sociality × fairness after collapsing across endowment. Two separate endowment regressors retain the high/low factor for nuisance control and a secondary high-minus-low contrast; they are not used for model selection or success criteria. They start at offer/decision onset and end at canonical choice-feedback end, following the existing decision endpoint. Two pre-offer regressors preserve sociality. Response impulses and mean-centered RT pmods use the authoritative convention, plus missed-trial and missed-feedback nuisance events when present. No orthogonalization, derivatives or high-pass filtering are introduced; TR, BOLD, 5 mm smoothing and TEDANA-plus-confounds match completed authoritative UGR model 3.

Before new data are inspected, the conservative design gate is fixed at: at least three valid trials per primary sociality × fairness cell and per endowment level, fair/unfair ratio at most 4:1 within sociality after collapsing across endowment, full rank among nonzero centered columns, condition number ≤1e8, absolute categorical EV correlation ≤.95 and categorical EV VIF ≤100. Report relative contrast efficiency as a diagnostic rather than a power guarantee. These thresholds stop the exploratory branch; they do not justify silently changing its scientific definition. The actual convolved `feat_model` matrix includes the original nuisance confounds.

Eight L1/L2 contrasts estimate the four collapsed primary conditions, social and nonsocial fairness effects, their interaction, and the secondary high-minus-low endowment effect. Both runs must complete fixed-effects L2. FEAT plots and subject-level counts/logs stay private; the first design preview has no participant identifier.

## Parallelism and outputs

Seven primary new transfer endpoints have 500 participant-label permutations each (when all branches are available). One flip per participant is shared across maps/tasks within a contrast family, including test labels. Each job refits all five folds. The process pool defaults to 96, bounded by CPU affinity and a conservative RAM estimate, with one BLAS/OpenMP thread per process. Read-only feature memmaps are shared by the OS; parent-only atomic indexed checkpoints allow changing worker count without changing random labels or repeating completed jobs. FEAT uses a separate conservative 8 GiB/job RAM budget and `FSLSUB_PARALLEL=1` to avoid nested oversubscription.

Commit only public aggregate results, sanitized development-derived maps/figures, reports and provenance. Never commit `work/`, membership, BIDS events, subject-level expressions, source-image paths or FEAT outputs. The collaborator email is **drafted only after results are available**, never sent. Missing branches remain explicitly identified; placeholder figures never imply chance performance.

Local checks exercise arithmetic, guards, actual-design mathematics, serial/parallel permutation identity, checkpoint resume and synthetic report/figure generation. This machine has no production private manifests or participant maps; real execution and participant design preview must occur on Linux2. The installed local FSL can additionally check synthetic rendered designs. Local synthetic checks are not scientific results.

## Local implementation validation

The new tests cover exact contrast arithmetic, matched samples, protected-person guards, current QC flags, nominal dollar-offer mapping, offer timing, design failure gates, fold/final-model restart, serial versus parallel indexed permutations, output isolation and orchestration with a stopped categorical branch. Synthetic figure rendering exercises all six requested figures and the 500–700-word email template.

The installed-FSL synthetic check now exercises the collapsed fairness/endowment design and eight expected contrasts. It verifies the rendered convolved matrix, contrast rows, rank, condition number, categorical EV correlation and VIF gates; deliberately empty missed-event columns remain nuisance-only. These are software/design checks on invented data, not participant results. The initial numeric-field quoting issue exposed by this check was corrected and regression-tested.

## October 5 completed phase-input handoff

The source is `rf1-sra-sharedreward` commit `d5331d38c298023eeaf094e6e83ee0df65434ac2`, model commit `562f0d15e0ac370a69ea36d40fdcb23f5e20f597`, audit `logs/records/phase-resolved-20261005-144746`. The reviewed upstream changes support empty neutral events, numeric FSL header whitespace and bounded worker threads; the 14 EVs and 34 canonical contrast definitions are unchanged. Updated source hashes reflect this reviewed execution contract.

The classifier uses only activation COPEs 1–6: C_pun, C_rew, F_pun, F_rew, S_pun, S_rew. It recomputes the existing equal-weight partner/context maps from these conditions. Cooper's crosswalk moves full-trial punishment differences 27/28 to phase-specific 31/32, but these precomputed differences are not classifier inputs. There is no decision-contrast or PPI input. The aging full-trial maps remain the frozen baseline/comparison; the new outcome-only models are development follow-ups, not replacements for the frozen candidates.

Cooper's behavioral eligibility and future project-specific imaging exclusions do not silently modify the classifier cohort. This adapter consumes the completed RF1 technical inventory plus our existing QC policy. Any later source-manifest/output change requires another provenance review. No preprocessing or Shared Reward L1/L2 refit is launched here.

After pulling the classifier update, run the dry run and then `--fairness-preview` again. The latter archives an older design-only preview when the source/code fingerprint changes and prints the fresh phase inventory. If it passes with a usable SR outcome sample, run `bash code/run_final_stress_tests.sh --workers 96`. Private Linux2 data are required for those commands; local checks do not establish the post-QC N or new accuracies.
