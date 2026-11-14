# Paired voxelwise Social Doors outcome effects

This bounded development analysis uses the exact existing N=178 three-paradigm cohort under tSNR/coverage/mean-FD QC. The primary N=50 validation cohort and all 62 protected participants remain excluded from image access. No new splits, classifier fits, model choices or validation scores.

On Linux2, in the existing analysis Python environment:

```bash
cd /ZPOOL/data/projects/rf1-sra-neural-signature
git pull --ff-only &&
  bash code/run_doors_voxelwise.sh --dry-run &&
  bash code/run_doors_voxelwise.sh --threads 16
```

The dry run checks original roster/split fingerprints, mask provenance, the retained participants' Doors QC, fitted FEAT contrast labels/weights and estimability, and source headers. It reads no participant voxels. Missing or changed inputs stop the run; no participant is replaced. The full run checks raw source means/norms against the original development source log. It reads only run-1 Social Doors and monetary Doors COPEs 1 and 2 and their FEAT masks; it does not read other task voxels. BOLD is used for header geometry only.

The launch uses the **original** QC table that defined N=178, verified against the frozen sample's SHA256. If the live table differs, it reads the immutable blob from upstream commit `5e8bbd62f2ed97bf21141f9995dbfa6b97e718f1` in the local `rf1-sra-linux2` checkout. The live table is diagnostic only: the launcher prints aggregate counts of new outliers, missing/ambiguous rows and unknown flags, and records them in `provenance/revised/doors_voxelwise/qc_audit.json` after a full-run preflight. No participant identifiers are written. This is an analysis of the original cohort, not a claim that the cohort passes a newer QC policy. If the historical Git object is missing, fetch it without checking out a different version:

```bash
git -C /ZPOOL/data/projects/rf1-sra-linux2 fetch origin 5e8bbd62f2ed97bf21141f9995dbfa6b97e718f1
```

| Test | Participant-level contrast | Alternative |
|---|---|---|
| Win | Social Doors COPE 1 − monetary Doors COPE 1 | Positive |
| Loss | Social Doors COPE 2 − monetary Doors COPE 2 | Positive |
| Social context × outcome valence | Win difference − loss difference | Two-sided |

Subtract uncentered COPEs in the common MNI grid, with the original linear interpolation for mismatched grids. Start from the frozen 84,045-voxel mask and retain only voxels with paired FEAT coverage for all 178 participants. For interpolated maps, coverage must include the interpolating neighborhood. This avoids interpreting an uncovered voxel as a zero-valued effect. The final coverage mask and voxel count are reported; the frozen mask is unchanged.

The paired group t-test is a one-sample t-test on the differences (df=177). Use 10,000 seeded participant sign flips, a whole-mask maximum t statistic for each simple effect, and maximum absolute t for the interaction. Bonferroni-adjust the three voxelwise FWE p maps as one family (alpha .05). P maps store p, not 1−p; outside-mask p is 1. Sign flipping assumes independent participants and symmetric difference errors under the null. See [FSL's paired inference guidance](https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/randomise.html) for the equivalent within-subject difference formulation.

The interaction is the direct win–loss comparison. A significant win effect and a nonsignificant loss effect do not establish an interaction, and a nonsignificant interaction does not establish equivalence. The simple effects compare different task runs and can include run, baseline or other task differences. COPE differences are not classification accuracies. The same outcome maps were already examined by decoders, so treat this as a development follow-up.

Open `reports/revised/doors_voxelwise/REPORT.md` after completion. Group mean COPE, t, corrected p, thresholded t and coverage maps are in `results/revised/doors_voxelwise/maps/`. The figure shows fixed slices and corrected significance outlines. The summary and significant-component peaks are in `results/revised/doors_voxelwise/aggregate/`; component sizes are descriptive, not cluster-level tests. `provenance/revised/doors_voxelwise/run_status.json` records completion and aggregate input fingerprints. Review those output directories before committing results. No participant maps, identities or scores are exported.

Local validation uses synthetic paired maps, brute-force sign flips, a protected-source rejection check and synthetic NIfTI/report rendering. Scientific results require the Linux2 run.
