# Implementation source audit

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

All requested COPE numbers and substantive definitions agree with current
activation templates. Important operational distinctions:

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
