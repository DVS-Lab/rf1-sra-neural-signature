# Internal multitask holdout policy — v2

Exactly N=50 participants form the **internal multitask validation set**. They
must have complete required ses-01 Shared Reward, Trust, socialdoors, monetary
doors, and UGR activation maps. Selection from Shared Reward alone is prohibited.
The complete intersection must contain at least 250 participants before selection,
leaving at least 200 for development. Otherwise stop and report exact overlap Ns.

Selection precedes any voxel-level modeling and uses seed 20260928, age quantile
× Shared Reward FlipAngle stratification, and explicitly recorded sparse-stratum
fallbacks. No neural, behavioral, or phenotype outcome balances the split.
File existence, geometry headers, BIDS/acquisition metadata, age, and available sex
may be inspected for inventory and description. Holdout voxel arrays, masks,
pattern expression, performance, and model selection remain inaccessible.

Persist `work/splits/subject_split_v1.tsv` locally and back it up securely with
`provenance/subject_split_v1.json`. The filename is retained for infrastructure
compatibility, but the JSON must specify `cohort_definition: multitask_complete_v2`.
A legacy single-task split is not silently reused, even when membership matches.
A missing TSV, checksum mismatch, or changed eligible cohort stops execution.
`--regenerate-split` is an explicit dangerous override that archives prior membership
and metadata and warns prominently. Redrawing cannot erase prior neural exposure;
it is not permission to choose a more favorable cohort or result.

All loading/fitting/scoring boundaries require an explicit development list and
its locked-partition guard. Multitask fitting records the union of all participants
in every training paradigm. OOF scoring rejects anyone in that union; LOPO also
rejects any supposedly held-out paradigm in the model's training tasks. These
rules apply to reward, secondary context, UGR, reliability, and every cross-family
application. No holdout-scoring command is implemented.

After development is frozen and the confirmatory protocol specified, ideally
preregistered, a separately reviewed implementation may evaluate the common reward
signature across all three reward paradigms and UGR in these same 50 people once.
Only after that independent validation may an optional final distributable model
be refit on all eligible RF1 participants; it is distinct from the validated
frozen development model.
