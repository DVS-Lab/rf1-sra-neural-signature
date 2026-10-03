# Frozen v4 characterization

This is a read-only scientific characterization of the completed v4 results at
`f93a44b`. Original participants, folds, masks, contrasts, classifier settings and
QC thresholds are frozen. Primary development counts are 192 (partner pair) and
178 (three paradigms); the 50 qualifying primary validation participants are never
loaded or scored. No validation command or flag exists.

## Linux2 launch

From the existing production checkout/environment:

```bash
git pull --ff-only
python3 -m pytest -q
bash code/run_characterization.sh --dry-run
bash code/run_characterization.sh
```

Keep the original private `work/splits/` and `work/revised/` files. The dry run
requires them and checks the full recorded membership, classifier, mask/weight
hashes and OOF/public-accuracy correspondence. It reads no NIfTI images. It writes
only the audit report and the existing process lock. Missing evidence or genuine
leakage stops characterization; no automatic repair is attempted.

The full launch then reconstructs exact spatially centered DEVELOPMENT inputs
and verifies saved fold-map margins against private OOF margins, with absolute
`tolerance=1e-5` and relative `tolerance=2e-5` to accommodate saved float32 weights.
Every binary forced-choice classification must also agree exactly. This second
gate precedes all scientific map/figure outputs. Original upstream completed
designs and aging image-stat fingerprints are checked before source voxel access.

## Outputs and definitions

- `reports/revised/characterization/IMPLEMENTATION_AUDIT.md`: audit status and limitations.
- `reports/revised/characterization/REPORT.md`: generated scientific summary and figure links.
- `results/revised/characterization/maps/`: byte-identical copies of raw weights,
  Haufe patterns, mean differences and bootstrap summaries.
- `results/revised/characterization/figures/`: Figure 1–6 plus supplements; PNG at
  300 dpi and PDF/SVG charts with vector text.
- `results/revised/characterization/aggregate/`: spatial correlations, descriptive
  clusters, OOF summaries, QC comparisons, reliability and permutation diagnostics.
- `provenance/revised/characterization/run_status.json`: completion, immutable-v4
  verification, and `holdout_scored=false`.
- `work/revised/characterization/`: private feature caches, distributions, OOF
  condition scores, input fingerprints and checkpoints. Never commit this tree.

Seventeen final binary models are characterized: common plus three task-specific
models for social context/reward in the three-paradigm cohort, and common plus two
task-specific models for positive closeness, negative closeness and valence in the
partner pair. Closeness reward modulation is retained in performance/reliability
figures as a null comparison, without a full anatomical characterization.

Haufe = Cov(X, Xw+b) / Var(Xw+b) on exactly the model's spatially centered input
maps. No feature z-scoring occurs. Common mean differences average all included
tasks equally. Raw SVM maps are discriminative coefficients, not activation.
Haufe and mean-difference maps are descriptive, not causal or inferential.

The five common models receive 1,000 whole-participant bootstrap refits. Every
resampled person contributes all their tasks/classes. Full float32 weight and
Haufe distributions are stored privately, with mean, sample SD and positive/
negative proportions saved publicly. Values with sign proportion >=.975 are
shown as **bootstrap sign-stable for descriptive visualization**. The sign-stable
map displays the bootstrap mean; raw and Haufe signs are assessed separately.
Components use 6-neighbor connectivity, minimum 20 voxels, and report coordinates
and bootstrap-mean Haufe values without inferential p values. No atlas is fetched.

Two hundred participant-label permutations refit common context/reward, common
positive/negative closeness and valence, plus the three context LOPO models. A
single flip per person is shared across their tasks and applied to both training
and test labels. Each fit preserves the original folds, features, mask and
LinearSVC settings. Empirical p=(1+#null>=observed)/(1+B) is a diagnostic, not a
replacement inferential framework. Null means outside [.45,.55] trigger
`needs_review`; this is a transparent descriptive centering check, not a formal
test or a reason to tune the model.

## Restart and computational cost

The requested defaults entail 5,000 bootstrap fits and 8,000 permutation-CV fits;
expect substantial CPU time. Work proceeds serially with bounded feature/voxel
blocks and memory-mapped private arrays. The launcher defaults BLAS/OpenMP thread
counts to one unless already configured. No models are tuned for faster fitting.
Private feature caches and bootstrap distributions require several GB of space
on the production grid; keep additional room for plots and checkpoints.

Restart the same command after interruption. Completed bootstrap batches of ten
and every completed permutation are verified and reused. An interrupted batch may
repeat up to nine bootstrap fits. Once computation is complete, plotting can be
restarted independently:

```bash
bash code/run_characterization.sh --plots-only
```

Plotting failures preserve expensive work. Input/code/settings/software drift is
rejected; restore the matching environment/files rather than deleting locks.
There is no force-reset flag. If changing the characterization calculations is
scientifically necessary, review and version the checkpoint migration explicitly.

The report does not automatically name brain networks from coordinates. Review
the actual maps before interpreting perceptual versus broader anatomical
contributions. This limitation is stated, and no cherry-picked exclusion mask is
constructed. All final results are development characterization, not validation.

## Local implementation validation

The full repository regression suite passed 67 tests on 2026-10-03. Subsequent
characterization checks also cover corrupted OOF margins and unrelated private
image files; the seven-test characterization suite exercises exact Haufe algebra,
participant-level permutation logic, bootstrap restart/integrity, descriptive
cluster coordinates, holdout rejection, original-file preservation and a full
synthetic characterization with Figure 1–6 rendering. Main figures and slice
mosaics were visually reviewed; a reliability-title overlap was corrected.

A real-data local dry run stops before NIfTI access because the Linux2 private
manifest is absent. The checked-in characterization REPORT and implementation
audit are explicitly pending status records. Real Haufe/bootstrap/permutation
results and anatomical conclusions require the Linux2 launch and figure review.

Final follow-up: all seven characterization tests passed (94.64 seconds),
including both added protection tests. The optional Nilearn path was also rendered
on synthetic volumes for all three display thresholds. It uses signed projections
and a symmetric colorbar, following the
[Nilearn plotting API](https://nilearn.github.io/stable/modules/generated/nilearn.plotting.plot_glass_brain.html).
No atlas was downloaded.
