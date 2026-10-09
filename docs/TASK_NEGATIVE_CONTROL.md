# Final empirical task-negative control

Status: Linux2 computations completed; the plotting-only failure was recovered locally from the committed results (see below). This is the bounded follow-up to the completed October 9 specificity package (6dd5352), not a replacement of its outputs or candidates.

## Completed run: reporting-only recovery

The `a500929` result push contains all observed comparisons, 500 permutations for each primary variant, and all 1,000 new control permutation jobs. The original prediction reconstruction is exact. Linux2 then failed importing `nilearn.plotting`: that Nilearn installation requires Matplotlib ≥3.8 while Linux2 has 3.7.1.

`code/render_task_negative_results.py` finishes the report from the committed, hash-pinned aggregate tables and derived group maps, using Matplotlib/Nibabel directly. It does not access private inputs, fit models, regenerate bootstrap samples, or run permutations. Numerical results and the original calculation code/identity remain unchanged; upgrading the numerical environment is unnecessary. A fixed standard anatomical background is checked by SHA256. Rendering inputs are pinned in `config/task_negative_render.json`.

The completed report and PNG/PDF figure are committed. **No Linux2 analysis rerun is needed; pull the final artifacts.** The original `run_status.json` retains its historical plotting failure. The separate `render_status.json` records successful reporting recovery, the input/output hashes, and that no models or permutations were rerun. This preserves the failure record instead of presenting a new full-pipeline execution.

For exact report reconstruction only, the standalone renderer accepts `--background /path/to/MNI152_T1_2mm_brain.nii.gz`; the file must match the reviewed background hash. The old full-computation command below documents the original launch and is not needed for this completed run.

## Fixed analysis

Reference: `monetary_doors_decision_task_negative`. Exact N=178 and five folds from the completed specificity run; same QC, 95% development-coverage mask, outcome arrays, class orientation and C=1 voxel SVM. No protected participant image is loaded. Source repositories are read-only; no FEAT is launched.

For each fold, equally average training people's **original signed, uncentered monetary Doors COPE3**; take `minimum(mean, 0)`; only then spatially center and unit-normalize. Apply that direction to both training and test maps. Compare the existing outcome SVM, a fixed one-feature C=1 readout (primal solver, training labels only), and the original SVM refitted after removing the template projection. No scaling/tuning, new mask, alternative template or model-selection rule.

Primary: SR→Trust and Trust→SR human–computer. Secondary: bidirectional friend–stranger and SR/Trust→social-versus-monetary Doors. Each endpoint averages the same four train/test-valence correctness cells within person. Direction construction is independent of social labels; the one-feature readout is supervised. Doors secondaries share monetary task context with the reference and are not independent validation.

The descriptive phase-resolved Haufe comparator pools SR/Trust positive/negative human–computer maps on each training fold. A separate full-development version is for display only. Neither changes the frozen signatures. Spatial comparison with Yeo-7 label 7 is descriptive only. Negative fractions and magnitudes are computed **before** centering. A <1% negative-support flag is a limitation, not a feature-selection threshold; a wholly nonnegative or constant template stops execution.

## Contrast and implicit-baseline audit

The reviewed [source model](https://github.com/DVS-Lab/rf1-sra-socdoors/blob/e77700ddf09a2fe9ab7e6ce9b32bc298082d419e/templates/L1_task-doors_model-1_type-act.fsf) has EV1 win, EV2 loss, EV3 decision, EV4 missed decision; COPE3 = [0,0,1,0]. Decisions and outcomes are modeled separately. Double-gamma convolution, no derivatives/orthogonalization, and fixed event filtering are checked against every actual design.

The reviewed [presentation code](https://github.com/DVS-Lab/rf1-sra/blob/9b9cf69348adcd6ec0b19861b637f1a63d1129cb/stimuli/Scan-Social_Doors/social_doors.py) draws fixation during predecision/postdecision intervals and jittered ITIs (reference schedules 1.1–11.6 seconds). It therefore supports a decision-versus-implicit-fixation interpretation, not rest or a universal DMN signature. Runtime checks require 40 ordered decision/feedback pairs, positive interior fixation gaps, event coverage within the scan, exact canonical/fitted EV timing correspondence, correct fitted contrast weights, estimability, valid geometry and retained QC. Only interior gaps are counted as fixation; initial/final unused acquisition time is not assumed to be rest. Missed decisions are separately modeled; the presentation can record planned win/loss labels when showing a missed-response message. That limitation is reported without changing exclusions.

Source template, presentation and schedule hashes are recorded in `config/task_negative_control.json`; historical task deployment is not independently established by the source review. Actual signed finite COPEs are checked at execution. A failed audit stops, without dropping subjects or substituting maps. Existing original predictions must be reconstructed before new results are interpreted.

## EV serialization audit repair

The initial audit used a 0.0001 s absolute tolerance, which can reject the upstream converter's valid output. The pinned `code/BIDSto3col.sh` subtracts zero from each onset and prints the numeric result with awk `%s` (default six significant digits); durations are passed through as strings. For example, 123.456789 s becomes 123.457 s. The comparison now accepts either canonical precision or precisely that onset serialization, with durations and unit amplitudes unchanged. It does not broadly relax timing tolerances or allow unexplained shifts.

Every checked EV records its matching rule and maximum onset/duration/amplitude differences. Identifying details stay in ignored `work/revised/task_negative_control/ev_timing_audit.tsv`; the sanitized aggregate `provenance/revised/task_negative_control/ev_timing_audit.json` is written even when a timing mismatch stops the run. A reproduced formatting bug explains a possible cause of the first Linux2 audit failure; confirmation for the real cohort still requires the repaired dry run. The previous run stopped before creating a computational identity or fitting checkpoints, so no reset or checkpoint deletion is needed.

## Linux2 launch and restart

```bash
cd /ZPOOL/data/projects/rf1-sra-neural-signature
git pull --ff-only
tmux new-session -A -s neural_signature
bash code/run_task_negative_control.sh --dry-run
bash code/run_task_negative_control.sh --workers 96
```

Run the full command only after the dry run passes. Detach with Ctrl-b, then d. Reattach with `tmux attach -t neural_signature`. Repeat the full command to resume after interruption. No deletion/reset or checkpoint bypass is provided. Inputs, software, code, QC, folds, prior caches and prior public artifacts are fingerprinted; drift stops for investigation.

The dry run hashes **development** source bytes, reads NIfTI headers, and checks fitted events/designs; it does not materialize participant voxel arrays, establish deactivation plausibility or compute predictions. It writes a sanitized baseline audit in the new namespace only. The existing private specificity outcome cache and predictions must still be available on Linux2.

500 synchronized participant-label permutations for the two primary transfers only. Original outcome nulls are reused after reproducing two original replicates with the same frozen RNG/folds/order; the two new variants require 1,000 indexed jobs total. Workers are bounded by available CPU and RAM, with one BLAS thread each. The requested 96 may be capped for memory. Private observed fold and permutation checkpoints persist as they complete. No duplicate full exploratory matrix is run. Bootstrap differences use 10,000 matched participant resamples of fixed OOF predictions; these intervals omit refitting uncertainty. MaxT across six primary variant/endpoint combinations is a complete-null diagnostic, not a test of differences.

## Outputs and hard stop

All new outputs live under `*/revised/task_negative_control/`:

- `reports/.../REPORT.md`: one report, including source/spatial sanity, matched results, signed expression, interpretation and remaining decisions.
- `results/.../figures/task_negative_comparison.{png,pdf}`: raw signed decision mean, centered empirical direction, descriptive outcome Haufe; primary accuracies and paired changes.
- `results/.../aggregate/`: baseline timings, negative-component measures, spatial correlations, transfer/valence tables, expression differences, permutation diagnostics and benchmark reconstruction.
- `results/.../maps/`: labelled fold-training and `DEV_display` derived maps. They have different units; positive values in the centered direction do not imply above-baseline activation.
- `provenance/.../run_status.json`: records the original execution, including its preserved plotting failure. For the recovered report, see `render_status.json` with `status=complete`, unchanged numerical inputs and `holdout_scored=false`.
- Ignored `work/.../`: participant identities, signed raw sources/features, private provenance, fold predictions and resumable null jobs. Never commit these.

Commit only reviewed sanitized public outputs, report, figure and derived group maps after Linux2 finishes. No participant/source paths appear in aggregate tables. All pre-existing code/config/public outputs are hash-pinned and remain unchanged.

Then stop. Agree on the construct and final candidate/source-phase strategy, external dataset and contrast mapping, QC/coverage, confirmatory endpoints/multiplicity/success criteria, and the preregistered N=50 validation plan. Do not launch further templates, atlas searches, UGR/betrayal or holdout scoring.
