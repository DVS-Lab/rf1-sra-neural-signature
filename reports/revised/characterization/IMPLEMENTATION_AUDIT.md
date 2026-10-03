# Implementation and leakage audit

**PENDING: private membership files are unavailable in this local checkout. No real-data scientific outputs were generated.**

CHARACTERIZATION AUDIT STOP: private revised sample manifest unavailable

Static review found no evident supervised leakage; the complete runtime/private-file audit has not passed. This is not a clean audit certification.

- Frozen implementation/config hashes checked against the completed v4 code.
- Estimator: sklearn.svm.LinearSVC(C=1.0, dual=True, class_weight=None, max_iter=100000, tol=0.0001, random_state=20260928).
- Private manifests, original split/folds, per-model train/test membership, OOF records, source-load logs and mask-source logs are required. Every test participant must be absent from training across all tasks; all maps retain one participant fold. LOPO excludes the entire target paradigm.
- All original validation participants remain protected. Primary QC-qualified validation N=50 is checked; no validation scoring path is implemented.
- Per-map spatial centering is independent. No population scaling, feature selection, or hyperparameter search occurs.
- Frozen masks use 95% development coverage without condition labels. Coverage includes development participants later tested in CV: this is a fixed, label-free development mask, not a strictly training-fold-only mask. It is retained as requested; no validation data contributes.
- Saved coefficient/mask hashes and OOF aggregate accuracy must match. Reconstruction from float32 saved fold weights is additionally checked before scientific map/plot outputs, using the documented numerical tolerance.
- Source-load logs and code can establish the pipeline's recorded behavior, not unrecorded manual data access outside this pipeline.

holdout_scored = False
