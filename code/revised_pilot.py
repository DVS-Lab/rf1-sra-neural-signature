"""Reconstruct QC-qualified samples and rerun development analyses; never score holdout."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import sys
import yaml
from build_mask import build_mask
from pilot import run_lock
from preflight import preflight
from revised_design import COHORTS, VERSION, revised_config
from revised_samples import inventory, reconstruct
from revised_models import run_models
from revised_reporting import report
from utils import load_config, PipelineError, write_json, write_tsv, sha256


def settings(base):
    path = base.root/'config/revised_analysis.yaml'
    spec = yaml.safe_load(path.read_text())
    if (spec['version'] != VERSION or spec['primary_qc_policy'] != 'tsnr_coverage_fd'
            or spec['sensitivity_qc_policy'] != 'prior_four_metric_policy'
            or spec['include_new_development'] is not True or spec['holdout_mode'] != 'top_up_pair'
            or spec['holdout_target_primary'] != 50):
        raise PipelineError('Revised policy differs from approved 50-person primary holdout / three-metric QC contract')
    return spec, sha256(path)


def run(base, dry_run=False, sample_only=False):
    spec, config_hash = settings(base)
    policies = [spec['primary_qc_policy'], spec['sensitivity_qc_policy']]
    c = revised_config(base)
    # Shares the original launcher's lock: v3/v4 cannot overwrite inputs concurrently.
    with (nullcontext() if dry_run else run_lock(base)):
        print('Revised preflight: reward-condition contrasts; no UGR or decision-phase maps', flush=True)
        repos, contracts, versions = preflight(c)
        frame = inventory(c, write=not dry_run)
        samples, summary = reconstruct(base, frame, policies, spec['include_new_development'],
                                       write=not dry_run, holdout_mode=spec['holdout_mode'])
        if dry_run:
            print('REVISED DRY RUN PASSED: reconstructed metadata only; no membership saved or voxel access.', flush=True)
            return summary
        sample_c = revised_config(base, 'samples')
        manifest_hash = sha256(sample_c.output('work/manifest.tsv'))
        write_json(sample_c, 'provenance/revision.json', {'settings': spec, 'config_sha256': config_hash,
                   'software': versions, 'contrast_contracts': contracts, 'holdout_scored': False})
        write_tsv(sample_c, 'provenance/repositories.tsv', repos)
        if sample_only:
            print('SAMPLES LOCKED: QC-qualified validation supplement fixed; no voxel data read.', flush=True)
            return summary
        overall = {'run_id': datetime.now(timezone.utc).isoformat(), 'status': 'in_progress',
                   'holdout_scored': False, 'sample_manifest_sha256': manifest_hash,
                   'config_sha256': config_hash, 'completed_scopes': []}
        write_json(sample_c, 'provenance/run_status.json', overall)
        for (policy, cohort), (guard, folds) in samples.items():
            scoped = revised_config(base, f'{policy}/{cohort}')
            # The authoritative aging index was verified during preflight/inventory.
            scoped.aging_subjects, scoped.aging_provenance = c.aging_subjects, c.aging_provenance
            stage = 'mask'
            status = {'run_id': overall['run_id'], 'status': 'in_progress',
                      'policy': policy, 'cohort': cohort, 'holdout_scored': False,
                      'sample_manifest_sha256': manifest_hash, 'config_sha256': config_hash}
            write_json(scoped, 'provenance/run_status.json', status)
            try:
                print(f'Revised analysis: {policy} / {cohort}; development N={len(guard.development)}', flush=True)
                mask, reference = build_mask(scoped, guard, sorted(guard.development), COHORTS[cohort])
                stage = 'models'
                predictions, runs = run_models(scoped, cohort, folds, mask, reference, guard,
                                               versions, repos, manifest_hash)
                stage = 'report'
                report(scoped, predictions, runs, summary[(summary.policy == policy) & (summary.cohort == cohort)])
                write_json(scoped, 'provenance/run_status.json', {**status, 'status': 'complete'})
                overall['completed_scopes'].append([policy, cohort])
                write_json(sample_c, 'provenance/run_status.json', overall)
            except Exception:
                write_json(scoped, 'provenance/run_status.json', {**status, 'status': 'failed', 'stage': stage})
                write_json(sample_c, 'provenance/run_status.json', {**overall, 'status': 'failed',
                           'failed_scope': [policy, cohort], 'stage': stage})
                raise
        write_json(sample_c, 'provenance/run_status.json', {**overall, 'status': 'complete'})
        print('REVISED DEVELOPMENT RUN COMPLETE. See reports/revised/. Holdout remains unscored.', flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run', action='store_true', help='Check reconstruction without saving membership or reading voxels')
    group.add_argument('--sample-only', action='store_true', help='Lock samples and exit before voxel access')
    args = parser.parse_args(argv)
    try: run(load_config(args.config_dir), args.dry_run, args.sample_only)
    except (PipelineError, FileNotFoundError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
