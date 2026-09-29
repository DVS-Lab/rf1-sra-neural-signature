# Shared Reward zero-input investigation

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

## Evidence from sharedreward-aging

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
