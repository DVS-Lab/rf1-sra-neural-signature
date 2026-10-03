# Implementation source audit

## Active Shared Reward adapter (approved 2026-10-02)

The user approved using the completed RF1 activation outputs in
`sharedreward-aging`, retaining verified one-/two-run strategies and marking
Shared Reward phase-specific decision analyses unavailable. The active source
was inspected at `323902ca9dd11871d04219a2f89c40ddb3bfa5d3`. Its frozen audit is
`logs/records/full-analysis-20260930-203949-441196`: 346 RF1 activation subjects,
309 fixed-effects and 37 L1 passthrough; all six reward-condition maps verified.
These are source-audit counts, not the five-task intersection or final cohort.

The adapter reads the audit's final summary and candidate table, checks the
frozen L1/L2 manifest and contrast-table hashes, and selects only `dataset=rf1`,
`session=01`, `type=act`. Each candidate must match its canonical participant,
strategy, retained runs, contrast label and COPE/VARCOPE/mask path. Saved model
provenance, actual FSF/design.con reward vectors, smoothing, BOLD grid evidence,
completion files and required map headers are checked without voxel loading.
Recorded small-file input hashes and image-input size/mtime fingerprints are
checked for staleness, consistent with the upstream audit's provenance boundary.

The canonical fixed-effects path is:

```text
sharedreward-aging/derivatives/fsl/rf1/sub-<ID>/ses-01/
  L2_task-sharedreward_model-fulltrial_type-act_sm-6.gfeat/cope<K>.feat/stats/cope1.nii.gz
```

Single-run subjects use the manifest's retained L1
`L1_task-sharedreward_model-fulltrial_type-act_run-<R>_sm-6.feat/stats/cope<K>.nii.gz`.
No excluded run is loaded or imputed. The primary COPE set is 1–6; neutral maps
are outside the verified candidate set, and COPE27/28 are punishment comparisons,
not decision maps. This is a ten-EV/28-contrast full-trial model, not the earlier
14-EV/34-contrast phase-resolved model. FEAT smoothing remains zero on total-6-mm
BOLD. The original audited RF1-only source below remains a resource dependency;
its phase-resolved statistical outputs are no longer inputs to this pipeline.

## Historical phase-resolved implementation audit

The source GitHub checkouts were read during implementation. No Linux2 data were
available and no RF1 voxel analysis was performed. Exact SHAs and inspected file
hashes are in `provenance/implementation_sources.json`; these are implementation
references, not substitutes for each real run's `provenance/repositories.tsv`.

| Source | Inspected HEAD |
| --- | --- |
| rf1-sra-linux2 | c44dd97dfe0a778fb48113546a3e60d776be38c2 |
| rf1-sra-sharedreward | d716f84412902671b82adfa2a22c265c09c2de01 |
| rf1-sra-trust | bc731ddd143eea8737f42c22e7e334f5a9dc5532 |
| rf1-sra-socdoors | 95661049e0bd46b52b276e7364533b4e51843b69 |
| rf1-sra-ugr | 4b44a8db136487df3f68fccf014848f9e593cf8d |

Reviewed root/code READMEs, workflow audits, path configurations, current activation
FSFs, manifest/completion logic, the Shared Reward contrast crosswalk, the canonical
QC policy documentation, and UGR offer-centering implementation. Linux2 uses
`code/pipeline_common.sh`; it has no `code/project_config.sh` or
`code/WORKFLOW_AUDIT.md` at the inspected HEAD.

The original requested COPE numbers agreed with the then-selected phase-resolved
activation templates. Historical operational distinctions (superseded above for
Shared Reward):

- Shared Reward is the phase-resolved 14-EV/34-COPE model. The separate aging
  repository's pooled full-trial model is scientifically different and is not used.
- Shared Reward FEAT L1 smoothing is 0; input total target smoothness is 6 mm,
  reflected by `smTo-6`. The L2 template retains `smooth=5` as a higher-level GUI
  setting; it does not request another L1 smoothing step. Trust templates use a
  `SMOOTH` placeholder rendered to 5. The code validates these separately.
- The Shared Reward crosswalk still calls a C_neu temporal-filter anomaly pending;
  the newer WORKFLOW_AUDIT and actual template resolve it to 0. Current templates
  are authoritative; no alternate contrast is created.
- UGR copes 13/14 are sums of the two condition-demeaned offer regressors, not
  single direct-condition contrasts or averages. The exact nonzero weights are
  recorded in configuration.
- SocDoors L2 combines the two tasks and is intentionally forbidden here.
- The source-exclusion directory is outside the sibling project roots. Missing or
  unreadable access is a preflight failure, not a zero count.
- The upstream QC coverage target removes cerebellum/brainstem; this analysis uses
  the whole-brain TemplateFlow mask or its explicitly reported development coverage
  fallback. Non-mandatory imaging flags do not change eligibility.

Linux2 validation still must establish that completed FEAT outputs and saved
design evidence are present at these paths, the original MNI BOLD header paths
remain readable, and the canonical metadata/exclusion directory are accessible.
The dry run inventories those contracts without loading voxels or creating a split.

## Trans-task architecture revision

The same checked-out current templates were re-inspected for clean decision/context
contrasts: Trust cope1=`c_C`, cope2=`c_F`, cope3=`c_S` (EVs 1, 2, 3 respectively),
and separate socialdoors/doors cope3=`decision` (EV3). Shared Reward decision
copes remain 27/28/29 (EVs 12/13/14). Their labels, EV titles, and nonzero vectors
are now included in runtime contract validation. UGR constants remain broad epochs
and are not promoted to isolated decision contrasts. No source template changed.
