# Final development cross-valence follow-up

This is a separate, prespecified development analysis of social context and
friend–stranger context. Existing v4 and characterization artifacts, cohorts,
QC, five participant folds, masks and model settings are frozen. There is no
validation-scoring option. The primary partner-pair cohort is N=192; the
secondary three-paradigm cohort is N=178; primary validation N=50 remains locked.

## Linux2 launch

From `/ZPOOL/data/projects/rf1-sra-neural-signature`, in the existing environment:

```bash
git pull --ff-only
python3 -m pytest -q
bash code/run_cross_valence.sh --dry-run
bash code/run_cross_valence.sh --workers 96
```

Run inside the existing tmux session. `--workers` defaults to 96. The launcher
uses one OpenMP/OpenBLAS/MKL thread per process; workers also enforce one thread
with threadpoolctl. Independent permutation jobs share one process pool across
all 11 training specifications, so scheduling does not wait for one model's 500
permutations before starting another model. Each job completes all five folds
for one training specification and one permutation index. Its label flips are
fixed by index and shared across tasks, valences, and constructs.

The requested 12 endpoints require 27,500 permutation fits: 11 training
specifications × 500 permutations × five folds. The two collapsed relational
endpoints share the same fitted model. Primary/secondary observed CV requires
105 fits, plus seven new final development fits. The original collapsed social
context final model is copied byte-for-byte, and its OOF predictions are reused.
No hyperparameters are changed to reduce runtime.

Read-only feature matrices use shared OS-backed NumPy memory maps. Fitting still
allocates per-worker training/solver arrays. Worker count is bounded by the
requested count, CPU affinity, and a conservative estimate using 80% of currently
available RAM. Thus 96 workers are used when the machine can accommodate them;
otherwise the launcher prints the limiting resources and effective worker count.
An estimate is not a measured peak-memory guarantee. Exact scheduling details are
saved in `provenance/revised/cross_valence/parallel_execution.json` and progress
in `permutation_progress.json`. Keep several GB of free disk space for private
feature caches and ample RAM for concurrent solver copies. Other server users'
resource use can affect the available budget.

## Restart

Rerun the same command after interruption. The parent process alone atomically
writes each completed permutation result under its explicit permutation index;
out-of-order worker completion cannot change random labels or result ordering.
Completed CV models and feature caches are also checked and reused. In-flight
jobs may repeat, but completed jobs are retained. Worker count may change on
restart without invalidating scientific checkpoints.

After all numerical work finishes, plotting can be retried without refitting:

```bash
bash code/run_cross_valence.sh --plots-only
```

Original inputs, source code, scientific settings and software versions are
fingerprinted. Drift is rejected; do not delete checkpoint files to bypass it.
The shared pipeline lock prevents concurrent v4/characterization/follow-up runs
in one checkout. There is no reset, validation, or visual-exclusion execution flag.

## Inputs and representations

All rows are participant-indexed, and every task and outcome from each person
uses the same original fold. Labels are +1 social/friend and -1 nonsocial/stranger.
Every resulting map is spatially mean-centered independently inside its original
cohort mask. No population normalization, feature selection, PCA or tuning occurs.

| Task | Positive COPEs, C/F/S | Negative COPEs, C/F/S |
| --- | --- | --- |
| Shared Reward, full trial | 2/4/6 (rew) | 1/3/5 (pun) |
| Trust, outcome | 5/7/9 (rec) | 4/6/8 (def) |

Positive social context is mean(F positive, S positive) versus C positive;
negative context uses the corresponding negative maps. Friend–stranger domains
retain F versus S separately at each valence. Collapsed friend–stranger context
averages each partner's positive and negative maps before centering. Doors uses
Social Doors versus monetary Doors COPE1 win and COPE2 loss; it does not affect
primary model selection. The aging one-/two-run source strategy is retained.

Social reward and closeness reward remain their frozen interaction-like
comparators, not synonyms for positive-outcome social/friend discrimination.
No context-to-interaction generalization criterion is introduced.

The primary four-domain matrices are ordered SR positive, SR negative, Trust
positive, Trust negative. The secondary six-domain matrix adds Doors positive
and Doors negative. Pooled positive/negative models are provided for both
constructs, including friend–stranger to support Figure D. Only the 12 specified
endpoints receive permutation checks. No new alternate-QC models are run.

## Statistics

Single-task accuracy is the fraction of participants with positive paired
margin, with ties incorrect. Its descriptive 95% interval is the same exact
binomial interval used in v4. Mean-margin intervals use 10,000 participant
bootstrap draws. Combined-task accuracy first averages the two task correctness
indicators within each person; its accuracy and mean-margin intervals bootstrap
whole participants, keeping both tasks together. It is not computed by testing
the sign of an average margin and does not use 2N independent observations.

The 500 label permutations use one sign per person, shared by all their training
and test maps. Diagnostic upper-tail p=(1+#null>=observed)/501. No confirmatory
multiplicity framework is implied. Null means outside [.45,.55] produce
`needs_review`. Spatial correlations and Haufe/mean maps are descriptive. No new
spatial bootstrap was requested; optional atlas labels apply to the existing
characterization's bootstrap-stable peaks and say so explicitly.

## Local anatomy and proposed visual control

The full run requires an already-installed standard T1 background, searched in
TemplateFlow's configured MNI152NLin6Asym directory and existing FSL roots
(including FSLDIR). If it cannot be located, set FSLDIR to an existing installation;
no download occurs. The exact chosen path and SHA256 are recorded. Anatomical
backgrounds are resampled only for display. Fixed slice locations are sagittal
x=0, coronal y=-50 and axial z=20 mm. Charts/matrices use 300-dpi PNG plus PDF/SVG.

Available local FSL XML atlas definitions are inventoried. Harvard-Oxford
cortical/subcortical maxprob-thr25-2mm maps, if installed, supply descriptive peak
labels. Their XML labels, installed version and image/label hashes are recorded.
A label describes the peak only, not an entire large component.

The sole proposed exclusion is the union of eight predeclared Harvard-Oxford
cortical labels: intracalcarine, supracalcarine, cuneal, lingual, occipital fusiform,
occipital pole, and inferior/superior lateral occipital cortex. The proposal lists
exact integer values, source identity and native/intersection voxel counts after
nearest-neighbor resampling. It is a broad anatomical occipital definition, not
a complete functional visual network. If this asset is unavailable, the report
marks the proposal unresolved; no result-dependent substitute is selected.

**Only the definition and voxel counts are prepared. No excluded-feature model
is fit. Review and approve the proposal prospectively before any sensitivity run.**

## Output locations

- `results/revised/cross_valence/aggregate/`: all performance cells, three matrices,
  original-OOF reconstruction checks, spatial similarities, interaction comparators,
  descriptive labels and 12 permutation summaries.
- `results/revised/cross_valence/maps/`: common positive/negative/collapsed raw
  weights, Haufe and mean maps; collapsed relational task weights; original mask.
- `results/revised/cross_valence/figures/`: Figures A–E and supplements.
- `reports/revised/cross_valence/`: REPORT, IMPLEMENTATION_AUDIT,
  FREEZE_CANDIDATES, VISUAL_SENSITIVITY_PROPOSAL.
- `provenance/revised/cross_valence/`: status, software/code fingerprints, models,
  candidate manifests, standard anatomy, proposal and parallel execution details.
- `work/revised/cross_valence/`: private feature caches, participant OOF margins,
  exact train/test memberships, fold coefficients, permutation checkpoints.
  Never commit this private tree.

Candidate manifests preserve exact inputs, weight/mask hashes, intercepts,
preprocessing and classifier settings. They are decision aids for preregistration,
not validated models. All 1,186 existing public artifacts are frozen against the
checked-in `config/cross_valence_frozen_outputs.json`, including the completed
characterization and scientific review. Protected validation remains unscored.

## Implementation validation

The complete repository suite passed **78 tests** (685.71 seconds). The new tests
cover exact representation arithmetic and orientation, all planned domain pairs,
participant/fold separation, pooled-map selection, combined-task statistics,
holdout rejection before image/model access, preservation of original artifacts,
full synthetic two-cohort execution, candidate hashes, local anatomical proposals,
and plotting restart without repeated fits. Serial and two-process permutation
outputs match exactly, including recovery of a missing indexed checkpoint result.
Follow-up focused checks also passed after hardening standard-anatomy symlink
rejection. Figures A, D and E were inspected on the synthetic run for layout.

The real-data local dry run stops before images because the private Linux2
manifest is unavailable here. The committed cross-valence reports explicitly
mark production results/audit/candidate selection as pending. No synthetic
figures are published as scientific results. A fresh comparison confirmed all
1,186 original frozen public artifacts unchanged.
