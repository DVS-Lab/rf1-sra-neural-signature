# Shared Reward zero-input investigation

## Resolution: use the existing aging outputs

On 2026-10-02 the user approved adapting to verified RF1 full-trial activation in
`sharedreward-aging`, preserving its one-/two-run subject strategies and marking
Shared Reward phase-specific decision analyses unavailable. The earlier proposal
to require new phase-resolved FEAT models is superseded. No upstream processing
is needed for this adapter. The old reader searched the RF1-only model directory;
it never searched the completed aging model location.

The active adapter uses `derivatives/fsl/rf1/sub-<ID>/ses-01/` within
`sharedreward-aging`, with `model-fulltrial_type-act_sm-6` L2 or the verified
retained L1. All six reward-condition maps are verified for 346 RF1 participants
in the frozen audit (309 two-run and 37 one-run). The five-task intersection must
still be established by Linux2 inventory and meet N>=250 before selecting N=50.
The primary three-paradigm reward design is retained with explicit full-trial
timing; secondary decision analyses now use Trust/Doors only. Neutral probes are
unavailable because those maps are outside the verified upstream candidate set.

See [active source contract](UPSTREAM_CONTRACT_AUDIT.md) and
[analysis plan](ANALYSIS_PLAN.md). The chronology below records the investigation
before this approved change and is not the current input requirement.

## Recheck on 2026-10-02

The pooled-model execution gap is now resolved. At `sharedreward-aging` revision
`323902ca9dd11871d04219a2f89c40ddb3bfa5d3`, the
[September 30 final audit](https://github.com/DVS-Lab/sharedreward-aging/blob/323902ca9dd11871d04219a2f89c40ddb3bfa5d3/logs/records/full-analysis-20260930-203949-441196/final/summary.json)
has `computational_gate_passed=true`: all 655 RF1 activation L1s and 346 RF1
activation subject outputs are verified. Across both datasets and activation/PPI,
all 1,488 L1s and 786 subject outputs are verified. Subject outputs include one-run
L1 passthrough; they are not all two-run L2s. The September 15 missing-model counts
below are historical and must no longer be presented as current pooled status.

This does **not yet establish readiness for the present neural-signature model**:

- The completed launcher fits the pooled full-trial, 28-contrast model under
  `sharedreward-aging`. This pipeline requires the RF1-only phase-resolved,
  34-contrast model under `rf1-sra-sharedreward`, including separate decision maps.
- The [pooled contrast table](https://github.com/DVS-Lab/sharedreward-aging/blob/323902ca9dd11871d04219a2f89c40ddb3bfa5d3/templates/FULLTRIAL_CONTRAST_CANDIDATE.tsv)
  makes the mismatch concrete: activation COPE27 is `F-S (pun)` and COPE28 is
  `F-C (pun)`, whereas this pipeline requires `F_dec`, `S_dec`, and `C_dec` at
  COPE27/28/29. Pooled activation has no COPE29. Even identically named reward
  conditions use full-trial rather than isolated outcome epochs.
- `rf1-sra-sharedreward` main remains `d716f84`; no newly tracked phase-resolved
  completion audit was found. Unchanged source Git history does not establish
  whether untracked models have since been fitted on Linux2.
- The neural-signature repository still contains the failed September 29 inventory,
  not a newer successful inventory. Its N=264 four-task overlap is historical;
  neither N=346 pooled RF1 outputs nor the aging project's behavioral cohort proves
  that at least 250 participants meet this pipeline's five-task requirements.

The [updated upstream status](https://github.com/DVS-Lab/sharedreward-aging/blob/323902ca9dd11871d04219a2f89c40ddb3bfa5d3/docs/CURRENT_STATUS.md)
also records the source resolutions as propagated; those resolved review requests
should not be repeated. Its behavioral eligibility policy belongs to that analysis
and is not silently imported into this neural-signature cohort.

The next decisive check is the header-only neural-signature dry run on Linux2,
using the command below. If the phase-resolved paths are still absent, their
upstream availability remains the prerequisite. A deliberate full-trial adaptation
would change the scientific model and cannot provide the currently specified
Shared Reward decision/context estimates. No path substitution, cohort-threshold
change, source execution, or holdout operation was made during this recheck.

## Original September 29 investigation

The Linux2 run `20260929T031742.288253Z`, committed in `d238884`, reported
`missing_feat_directory` for Shared Reward L1 run 1, L1 run 2, and L2 for every
one of the 352 non-source-excluded BIDS participants. The failure occurred before
Shared Reward image headers, completion markers, or design checks. Thus it does
not establish that Shared Reward scans failed QC or that no Shared Reward data
exist elsewhere.

The inventory expects these directories under
`/ZPOOL/data/projects/rf1-sra-sharedreward/derivatives/fsl/sub-<ID>/ses-01/`:

```text
L1_task-sharedreward_ses-01_model-1_type-act_run-1_smTo-6.feat
L1_task-sharedreward_ses-01_model-1_type-act_run-2_smTo-6.feat
L2_task-sharedreward_ses-01_model-1_type-act_smTo-6.gfeat
```

These match `l1_output_base` and `l2_output_base` in upstream
`code/project_config.sh` at `670571339f97bc831cd6e4fd17a7cff42b9315a6`, the Shared
Reward revision recorded by the failed run, with its default derivative root and
6-mm target. They also match the reviewed revision `d716f84`. The production source
checkout was marked dirty, so committed source alone cannot establish its local
settings or which derivatives were actually generated. The downstream pipeline
does not inherit upstream `FSL_DERIVATIVES_ROOT` or `TARGET_FWHM_MM` overrides.

The remaining possibilities include unavailable/unmounted derivatives, a custom
output location, a different naming/model version, or an authoritative activation
run that has not yet completed. Distinguishing these requires filesystem evidence
from Linux2; the development machine has no access to it. One existing Shared
Reward L2 directory path is a useful starting point. If the directory is found,
its saved L1/L2 designs must still establish the intended phase-resolved model.
Historical or pooled-aging maps must not be substituted based only on task names.

## Historical September 15 evidence from sharedreward-aging

At the user's request, `DVS-Lab/sharedreward-aging` was also inspected at
`79ee586bbef71821af1079cf714ecb98f2ef7d61`. Its
[current status](https://github.com/DVS-Lab/sharedreward-aging/blob/79ee586bbef71821af1079cf714ecb98f2ef7d61/docs/CURRENT_STATUS.md)
and [September 15 audit](https://github.com/DVS-Lab/sharedreward-aging/blob/79ee586bbef71821af1079cf714ecb98f2ef7d61/logs/records/group-readiness-20260915-103331/summary.json)
distinguish prepared inputs from completed models:

- 655 RF1 runs were task-ready for the pooled model, but all 655 expected RF1
  activation L1s were missing at their canonical pooled paths. All 346 RF1
  subject-session activation candidates were unverified.
- Across both datasets and activation/PPI types, only four L1s and two subject
  outputs were verified, all for the isolated ds003745 pilot. The other 1,484
  expected L1s were absent. The status explicitly calls full-cohort execution
  outstanding; the smoothed-BOLD and input-QC inventories do not certify FEAT.
- Pooled outputs belong in `sharedreward-aging/derivatives/fsl/{dataset}/...`
  and use a different full-trial model with 28 activation contrasts. The neural
  signature requires RF1-only phase-resolved 14-EV/34-contrast activation maps,
  including separate decision contrasts. A path redirect cannot reconcile these
  scientific contracts, even if pooled models have subsequently completed.

Together with the September 29 neural-signature inventory and the user's
confirmation that the expected directories do not exist, this supports missing
upstream model execution/availability as the likely blocker. The September 15
pooled audit is not a fresh filesystem audit of RF1-only models and cannot prove
that no compatible outputs exist at custom locations. Before launching the neural
signature, establish the availability of the RF1-only phase-resolved L1 runs and
their two-run fixed-effects L2 outputs. If these have never been fitted, their
execution is a separate upstream prerequisite. This downstream change does not
run FEAT, redo preprocessing/smoothing, modify either source project, or substitute
the aging model.

## Diagnostic rerun

After pulling the diagnostic update, run:

```bash
bash code/run_pilot.sh --dry-run
```

The inventory now prints reasons, expected directory counts, and recognized
alternate layouts from the Shared Reward and canonical Linux2 FSL trees. It reads
directory entries only for this added diagnosis and excludes source-excluded
participants. No participant IDs or arbitrary filenames enter the aggregate layout
report. The search is intentionally bounded to these trees; zero counts cannot
rule out custom locations elsewhere. A real launch also saves
`results/aggregate/feat_path_summary.tsv` before the cohort gate.

The other four tasks intersect at N=264. This is an upper bound on the eventual
five-task cohort, not evidence that all 264 have usable Shared Reward maps. At
least 250 of them must have valid Shared Reward inputs to satisfy the approved
threshold and retain at least 200 development participants after reserving 50.
The failed run stopped before creating a split or loading modeling voxels. The
250-person gate, holdout policy, and map-selection contracts remain unchanged.
