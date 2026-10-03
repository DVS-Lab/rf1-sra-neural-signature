# Internal multitask holdout policy — v3

Exactly N=50 participants form the **internal multitask validation set**. They
must have complete required ses-01 Shared Reward, Trust, socialdoors, monetary
doors, and UGR activation maps. Selection from Shared Reward alone is prohibited.
The complete intersection must contain at least 250 participants before selection,
leaving at least 200 for development. Otherwise stop and report exact overlap Ns.

Shared Reward completeness is defined by verified RF1 full-trial activation and
its retained one-/two-run strategy. Neither missing decision-phase maps nor a
single retained run disqualifies an otherwise complete participant. Reliability
uses only the two-run development subset.

Selection precedes any voxel-level modeling and uses seed 20260928, age quantile
× Shared Reward FlipAngle stratification, and explicitly recorded sparse-stratum
fallbacks. No neural, behavioral, or phenotype outcome balances the split.
FlipAngle uses the verified retained Shared Reward run(s), supports BIDS inheritance
and equivalent integer indices (`run-1` / `run-01`), and checks all observed ses-01
Shared Reward run/echo values for agreement. An absent unretained run is not missing
metadata. Per the October 3 authorization to recover missing acquisition groups,
a missing retained run/echo value may use the unanimous recorded value from the
same participant's same-session Shared Reward acquisition. Such recovery is flagged
`flipangle_recovered_within_subject` in the private inventory and counted in public
metadata summaries. No value is inferred from age, neighboring participant IDs,
other participants, neural data, or the mere existence of two possible groups.
If no recorded value exists, or recorded values conflict, the status remains
missing/discrepant; the pipeline does not silently choose a group or relax the
holdout stratification policy. Existing locked holdout membership is always reused.

File existence, geometry headers, BIDS/acquisition metadata, age, and available sex
may be inspected for inventory and description. Holdout voxel arrays, masks,
pattern expression, performance, and model selection remain inaccessible.

Persist `work/splits/subject_split_v1.tsv` locally and back it up securely with
`provenance/subject_split_v1.json`. The filename is retained for infrastructure
compatibility, but the JSON must specify `cohort_definition: multitask_complete_v3_aging_fulltrial`.
A legacy single-task or phase-resolved v2 split is not silently reused, even when membership matches.
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
