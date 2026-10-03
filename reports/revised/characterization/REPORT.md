# Characterization status — Linux2 execution pending

This is a status record, not a completed scientific characterization report.

The characterization implementation has been exercised on synthetic development
data. The real-data launch in this checkout stopped before image access because
the private revised sample manifest is available only on Linux2. The complete
membership/leakage audit, numerical OOF check, Haufe maps, bootstrap distributions,
permutation diagnostics, and anatomical interpretation are therefore pending.
See [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md).

No leakage problem has been established from the static code review. A missing
private manifest does not constitute evidence of leakage, and it does not permit
certifying that the complete audit passed.

All 645 tracked pre-characterization result/report/provenance artifacts were
compared byte-for-byte with completed v4 commit `f93a44b`; none changed. No
holdout images were loaded, no participant assignments changed, and no validation
scoring option was added. `holdout_scored = False`.

Run on Linux2 from the production checkout, preserving its private files:

```bash
git pull --ff-only
python3 -m pytest -q
bash code/run_characterization.sh --dry-run
bash code/run_characterization.sh
```

The full launch replaces this status page with the evidence-based report only
after audit and OOF reconstruction pass. It prints Figure 1–6 paths, the spatial
correlation tables, and permutation diagnostics. Resampling is checkpointed;
`--plots-only` retries plotting after completed computation.

Implementation and methodological details:
[docs/CHARACTERIZATION.md](../../../docs/CHARACTERIZATION.md).
