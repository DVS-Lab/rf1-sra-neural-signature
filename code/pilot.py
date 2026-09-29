"""Single launcher orchestration. The only permitted neural cohort is development."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import sys
from build_mask import build_mask
from inventory import inventory
from make_split import make_split, make_folds, guard_from_split
from preflight import preflight
from reporting import report
from signature_pilot import run_models
from utils import load_config, PipelineError, write_tsv, write_json, sha256


@contextmanager
def run_lock(c):
    path = c.output('work/pilot.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise PipelineError('Another pilot is using this checkout') from exc
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def run(c, dry_run=False, regenerate_split=False):
    print('Preflight: paths, dependencies, current contrast templates, repository provenance', flush=True)
    repos, contracts, versions = preflight(c)
    if dry_run:
        print('Dry-run: full mechanical inventory using headers only; no output writes.', flush=True)
        inventory(c, write=False)
        print('Planned: locked N=50 split -> five development folds -> fixed mask -> OOF primary/cross-task/specificity/reliability -> final DEV model -> aggregate reports.', flush=True)
        print('DRY RUN PASSED. No split, voxel matrix, classifier, or result was created.', flush=True)
        return
    with run_lock(c):
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        write_tsv(c, 'provenance/repositories.tsv', repos)
        write_tsv(c, f'provenance/runs/{run_id}/repositories.tsv', repos)
        write_json(c, 'provenance/contrast_verification.json', contracts)
        write_json(c, 'provenance/run_status.json', {'run_id': run_id, 'status': 'in_progress'})
        try:
            print('Inventory: primary eligibility, secondary availability, metadata', flush=True)
            table, _ = inventory(c)
            eligible = table[table.eligible].copy()
            print('Holdout lock: selection/reuse before voxel access', flush=True)
            split = make_split(c, eligible, regenerate_split)
            guard = guard_from_split(split)
            development_subjects = sorted(guard.development)
            development = eligible[eligible.subject.isin(development_subjects)]
            folds = make_folds(c, development, guard, development_subjects,
                               sha256(c.output('work/splits/subject_split_v1.tsv')))
            print('Analysis mask: development data only', flush=True)
            mask, reference = build_mask(c, guard, development_subjects)
            print('Models: participant-blocked CV, OOF transfer, specificity, reliability', flush=True)
            primary, cross, weights, diagnostics = run_models(c, table, folds, mask, reference,
                                                             guard, development_subjects, versions, repos)
            print('Reporting: aggregate metrics and shareable figures', flush=True)
            report(c, table, split, primary, cross, weights, repos, diagnostics)
            write_json(c, 'provenance/run_status.json', {'run_id': run_id, 'status': 'complete', 'holdout_scored': False})
        except Exception:
            # No subject IDs or source-exclusion reasons in public failure status.
            write_json(c, 'provenance/run_status.json', {'run_id': run_id, 'status': 'failed',
                                                       'note': 'Outputs may be partial; rerun after resolving local diagnostic failure.'})
            raise
    print('PILOT COMPLETE. See reports/REPORT.md and reports/CODEX_REVIEW.md. Holdout remains locked.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', help='Alternate three-file configuration directory')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--regenerate-split', action='store_true', help='DANGEROUS: replace the locked split; invalidates independent validation if exposed')
    args = parser.parse_args(argv)
    if args.dry_run and args.regenerate_split:
        parser.error('--dry-run and --regenerate-split cannot be combined')
    try:
        run(load_config(args.config_dir), args.dry_run, args.regenerate_split)
    except (PipelineError, FileNotFoundError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
