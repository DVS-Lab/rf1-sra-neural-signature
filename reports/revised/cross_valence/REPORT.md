# Cross-valence follow-up — Linux2 execution pending

The implementation is prepared; these are not real-data results. The local dry
run stops at the required private revised sample manifest, which is available
only in the production Linux2 checkout. No local source or validation images
were loaded. The completed v4 and characterization results remain unchanged.

On Linux2, retain all private split, membership, QC and source records:

```bash
git pull --ff-only
python3 -m pytest -q
bash code/run_cross_valence.sh --dry-run
bash code/run_cross_valence.sh --workers 96
```

The production run will replace this page with both primary 4×4 matrices, pooled
cross-valence results, collapsed friend–stranger results, spatial comparisons,
12 prespecified permutation diagnostics and the six-domain Doors boundary matrix.
Figures use local standard anatomical backgrounds. See
[launch and restart documentation](../../../docs/CROSS_VALENCE.md).

The scheduler requests 96 independent permutation workers, one numerical-library
thread each, bounded by CPU affinity and estimated available RAM. Completed
permutation indices and fold models are checkpointed. Changing worker count on
restart does not change random labels or invalidate scientific checkpoints.

Validation N=50 remains protected. **holdout_scored=False.**
