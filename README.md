# RF1-SRA social modulation of reward signature

A downstream analysis of existing RF1 statistical images: can a distributed
whole-brain pattern distinguish reward effects in social versus computer contexts,
and generalize to Social Doors and Trust? This repository does **not** rerun
preprocessing, TEDANA, FEAT, or trial-level estimation.

Exactly **50 eligible participants are locked before voxel access**. All modeling,
mask construction, cross-task evaluation, and reliability analyses use only the
remaining development participants. No holdout scoring is implemented.

## Linux2 quick start

Use Python 3.10 or newer with the dependencies installed into your chosen environment:

```bash
cd /ZPOOL/data/projects/rf1-sra-neural-signature
python3 -m pip install -r requirements.txt
python3 -m pytest -q
bash code/run_pilot.sh --dry-run
bash code/run_pilot.sh
```

Set `PYTHON=/path/to/environment/bin/python` for the launcher if needed. A real run
executes preflight, repository provenance, header-only inventory, the persistent
N=50 split, five development folds, a fixed mask, primary CV, out-of-fold transfer,
specificity, reliability, the final development signature, and reporting.

**No real RF1 analysis was run during implementation.** The tests create only
synthetic images and temporary repositories and need neither `/ZPOOL` nor FSL.

## Read-only sources and configuration

The five sibling repositories default to `/ZPOOL/data/projects/`:

| Repository | Inputs |
| --- | --- |
| `rf1-sra-linux2` | Canonical BIDS, baseline participants.tsv, acquisition JSON, source-exclusion policy, descriptive QC |
| `rf1-sra-sharedreward` | Model 1 activation, smTo-6, L2 fixed effects and L1 runs 1/2 |
| `rf1-sra-trust` | Model 1 activation, sm-5, L2 fixed effects |
| `rf1-sra-socdoors` | Separate socialdoors and doors model 1 sm-5 L1 run-1 cope4; never their combined L2 |
| `rf1-sra-ugr` | Model 3 activation, sm-5, L2 fixed effects |

`config/paths.yaml` defines paths and metadata columns. Override `PROJECTS_ROOT`,
`RF1_LINUX2_ROOT`, `RF1_SHAREDREWARD_ROOT`, `RF1_TRUST_ROOT`, `RF1_SOCDOORS_ROOT`,
`RF1_UGR_ROOT`, `BIDS_ROOT`, `FMRIPREP_ROOT`, `SOURCEDATA_EXCLUSIONS_ROOT`, or
`TEMPLATEFLOW_HOME`. An alternate configuration directory can be supplied with
`--config-dir`. The project root always comes from this checkout's code location;
there is no arbitrary output-root option. All outputs must be inside this checkout
and disjoint from source trees, including resolved symlinks. Existing hard-linked
output files are refused. No upstream scripts are executed or repositories pulled.

The authoritative exclusion root must exist and be readable; a missing directory
is never interpreted as zero exclusions. Only `Smith-SRA-<ID>` directory existence
is inspected. Its contents and reasons are never read or published.

`config/contrast_spec.yaml` records actual nonzero contrast weights and names.
Every launch checks current activation templates; inventory also checks completed
`design.fsf` and `design.con` evidence. Ambiguous primary scientific contracts stop
the pipeline. Cross-task missingness is recorded without discarding otherwise
eligible Shared Reward participants.

## Restart and split preservation

Rerun the same command after interruption. Inventory and derived modeling products
are recomputed; the split and matching folds are reused and hash-checked. Only a
successful `provenance/run_status.json` with `status: complete` certifies a complete
set of outputs. An interrupted run can leave partial/stale products; do not publish
them. Writes are atomic per file and a process lock prevents concurrent launchers.

The local split TSV is irreplaceable study infrastructure: back it up securely,
together with its public JSON checksum. If the TSV disappears while its JSON
remains, the launcher refuses to redraw. Eligibility drift also stops the run.
`--regenerate-split` is an explicit **dangerous** override, prints a warning, and
archives old membership locally. It cannot undo prior neural exposure. Never use
it merely to improve balance or results.

## Outputs and privacy

| Location | Policy |
| --- | --- |
| `work/inventory/` | Private availability and header inventory; ignored |
| `work/splits/` | Private split and development folds; ignored; securely back up |
| `work/predictions/` | Private participant predictions; ignored |
| `work/diagnostics/` | Private norms, resampling records, missing-task reasons; ignored |
| `results/aggregate/` | Counts, demographics, performance, reliability, weight correlations; shareable |
| `results/maps/` | Fixed mask, five fold maps, final development weights; shareable |
| `results/figures/` | Aggregate accuracy/margins, unlabelled reliability scatter, stability; shareable |
| `provenance/` | Repository SHAs/paths, split/fold hashes and aggregate metadata, software/model settings; shareable |
| `reports/REPORT.md`, `reports/CODEX_REVIEW.md` | Aggregate development reports; shareable |

Participant identities and source-exclusion reasons must never be added to public
Git history, including with `git add -f`. The code deliberately retains only an
aggregate source-exclusion count. Shareable mask provenance points to a private
reference-path record rather than publishing the reference participant's ID.

See [the analysis plan](docs/ANALYSIS_PLAN.md), [holdout policy](docs/HOLDOUT_POLICY.md),
and [upstream contract audit](docs/UPSTREAM_CONTRACT_AUDIT.md). A future external
validation adapter can supply positive/negative maps, participant ID, optional run,
and explicit reference-space metadata to the documented common preprocessing and
scoring functions after an independent validation protocol is added.
