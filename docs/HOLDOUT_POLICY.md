# Internal holdout policy

The N=50 holdout is independent internal validation, not a development fold.
It is selected from mechanically eligible ses-01 Shared Reward participants,
before any voxel-level modeling, using seed 20260928. Age quantile × FlipAngle
stratification tries quintiles, quartiles, tertiles, then FlipAngle alone. Every
fallback is recorded. Unknown metadata are explicit categories, never imputed;
unusable strata cause a clear failure rather than an unannounced random split.

Only file existence, geometry headers, BIDS/acquisition metadata, age, and sex are
inspected for inventory and balance. Holdout voxel arrays, mask values, pattern
expression, predictions, and accuracy are not accessed. Inventory is header-only
for everyone, even before membership exists. Finite geometry is not a claim that
unread voxel values are finite. Development voxel validity is checked after the
split; primary nonfinite data stop the run without redrawing the holdout.

`work/splits/subject_split_v1.tsv` is private and must be securely backed up.
`provenance/subject_split_v1.json` records seed, exact method, aggregate balance,
and its SHA256. Existing membership is reused. Lost membership, hash mismatch,
or cohort drift stop processing. The explicit `--regenerate-split` override
archives old membership locally and warns prominently. Redrawing after development
has started can invalidate the claimed independence; the option is an operational
escape hatch, not scientific permission to redraw.

Every participant-data loading, fitting, and scoring entry point requires an
explicit development list and a guard derived from the locked partition. The
loader checks the subject before even opening the image; scoring additionally
rejects any participant who trained that fold model. There is no command or API
for applying the final model to holdout participants in this version.

After development methods and the signature have been frozen, ideally with a
preregistered confirmatory analysis, a separately reviewed implementation can
evaluate the holdout once. Only after that independent validation may a final
distributable model optionally be refit on all eligible RF1 participants. Such a
refit must not be presented as the independently validated development model.
