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

Restart with the same command. Feature caches, CV coefficients and indexed permutations are fingerprinted; completed jobs are reused. FEAT completion is checked explicitly because FSL can submit jobs internally. An existing incomplete categorical FEAT directory stops rather than being deleted or silently overwritten; inspect the private log/output before repair. Changed code, membership, QC or inputs requires review and a fresh explicitly managed checkpoint, never automatic mixing.

```bash
bash code/run_final_stress_tests.sh --plots-only
```

Plots-only requires completed, hash-verified computations. It does not refit classifiers or run FEAT. Do not edit frozen v4/cross-valence artifacts to satisfy a failed check.

## Samples, maps and evidence types

Only the frozen N=192 primary development pool can contribute; original folds and all original holdout assignments remain fixed. Current exclusions and canonical `tsnr_outlier`, `brain_coverage_outlier`, `fd_mean_outlier` flags determine downstream availability. No new IQR fences are estimated. Unknown QC fails. Baseline SR uses its previously retained aging runs; Trust and UGR L2 use runs 1+2. Phase-resolved SR uses the authoritative two-run L2 model-1 smTo-6 outputs. There is no silent aging or single-run fallback. Counts and missing reasons are printed **before fitting** and stored publicly in aggregate; membership stays private.

Phase matrices compare SR full-trial, SR outcome-only and Trust outcome on the identical intersection. Outcome-only SR+Trust pooled sensitivity models are separate artifacts. Original-architecture whole-brain and occipital-excluded models use a matched current-QC baseline subset. UGR transfers use source/target intersections, with the same person excluded across all tasks at test time. At least ten matched development people and all five original folds are required computationally; that is not a power guarantee.

Fixed final-DEV signatures applied to phase-resolved SR or UGR retain their exact weights. Those applications involve training-participant overlap and are **descriptive new-map probes**, not unseen-person validation. Their labels, tables and figures identify this explicitly. Generic valence uses original fold-specific coefficients, sign-reversed so positive corresponds to defection/unfairness, and is evaluated on the same norm-transfer people. Raw margins from different models do not have a shared scale.

The original analysis mask remains fixed. The exact eight Harvard-Oxford parcels and atlas/XML SHA256 values come from the approved cross-valence proposal; 9,760 of 84,133 primary-mask voxels are removed and 74,373 retained. Each map is recentered within the retained mask. Report original/excluded accuracy, difference, retained count and weight correlation on retained voxels; no success threshold is applied.

## Categorical UGR model

All outputs are under private `work/revised/final_stress_tests/fairness/`, named `model-signature-fairness`, and never inside upstream derivatives. Canonical BIDS only; no raw PsychoPy logs. The user approved nominal percentages with rounded dollars:

| Endowment | Unfair: nominal 5% / 10% | Fair: nominal 25% / 50% |
|---|---|---|
| $16 | $1 / $2 | $4 / $8 |
| $32 | $2 / $3 | $8 / $16 |

No unknown offer is rounded or reassigned after reading results. Every valid offer must match exactly one allowed amount. Eight categorical regressors preserve sociality × endowment × fairness. They start at offer/decision onset and end at canonical choice-feedback end, following the existing decision endpoint. Four pre-offer regressors preserve sociality/endowment separately. Response impulses and mean-centered RT pmods use the authoritative convention, plus missed-trial and missed-feedback nuisance events when present. No orthogonalization, derivatives or high-pass filtering are introduced; TR, BOLD, 5 mm smoothing and TEDANA-plus-confounds match completed authoritative UGR model 3.

Before new data are inspected, the conservative design gate is fixed at: at least three valid trials per categorical cell, fair/unfair ratio at most 4:1 within sociality/endowment, full rank among nonzero centered columns, condition number ≤1e8, absolute categorical EV correlation ≤.95 and categorical EV VIF ≤100. Report relative contrast efficiency as a diagnostic rather than a power guarantee. These thresholds stop the exploratory branch; they do not justify silently changing its scientific definition. The actual convolved `feat_model` matrix includes the original nuisance confounds.

Seven L1/L2 contrasts average high/low endowments equally: social unfair, social fair, nonsocial unfair, nonsocial fair, social unfair-minus-fair, nonsocial unfair-minus-fair, and the difference of those effects. Both runs must complete fixed-effects L2. FEAT plots and subject-level counts/logs stay private; the first design preview has no participant identifier.

## Parallelism and outputs

Seven primary new transfer endpoints have 500 participant-label permutations each (when all branches are available). One flip per participant is shared across maps/tasks within a contrast family, including test labels. Each job refits all five folds. The process pool defaults to 96, bounded by CPU affinity and a conservative RAM estimate, with one BLAS/OpenMP thread per process. Read-only feature memmaps are shared by the OS; parent-only atomic indexed checkpoints allow changing worker count without changing random labels or repeating completed jobs. FEAT uses a separate conservative 8 GiB/job RAM budget and `FSLSUB_PARALLEL=1` to avoid nested oversubscription.

Commit only public aggregate results, sanitized development-derived maps/figures, reports and provenance. Never commit `work/`, membership, BIDS events, subject-level expressions, source-image paths or FEAT outputs. The collaborator email is **drafted only after results are available**, never sent. Missing branches remain explicitly identified; placeholder figures never imply chance performance.

Local checks exercise arithmetic, guards, actual-design mathematics, serial/parallel permutation identity, checkpoint resume and synthetic report/figure generation. This machine has no production private manifests, participant maps or FSL installation; real execution and design preview must occur on Linux2. Local synthetic checks are not scientific results.
