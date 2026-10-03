"""Validate current upstream templates without executing any upstream code."""
import importlib.metadata
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from utils import PipelineError, sha256


def parse_fsf(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r'^\s*set\s+(\S+)\s+(.*?)\s*$', line)
        if match:
            key, value = match.groups()
            result[key] = value.strip('"')
    return result


def verify_l1(values, spec, rendered=False):
    if int(values.get('fmri(ncon_real)', -1)) != spec['n_copes']:
        raise PipelineError('Activation contrast count mismatch')
    smooth = '5' if rendered and spec['smoothing'] == 'SMOOTH' else spec['smoothing']
    if values.get('fmri(smooth)') != smooth or values.get('fmri(regstandard_yn)') != '0':
        raise PipelineError('Unexpected smoothing or registration contract')
    if spec['model'] == 'fulltrial':
        if any(values.get(f'fmri({key})') != '10' for key in ['evs_orig', 'evs_real']):
            raise PipelineError('Shared Reward must be the ten-EV full-trial activation design')
        if any(values.get(f'fmri(tempfilt_yn{i})') != '0' for i in range(1, 11)):
            raise PipelineError('Unexpected full-trial EV temporal filtering')
    for i, title in spec['ev_titles'].items():
        if values.get(f'fmri(evtitle{i})') != title:
            raise PipelineError('Activation EV ordering/title mismatch')
    for k, contract in spec['copes'].items():
        if values.get(f'fmri(conname_real.{k})') != contract['name']:
            raise PipelineError(f'Contrast label mismatch for cope{k}')
        actual = {int(m.group(1)): float(v) for key, v in values.items()
                  if (m := re.fullmatch(rf'fmri\(con_real{k}\.(\d+)\)', key)) and float(v) != 0}
        if actual != contract['weights']:
            raise PipelineError(f'Contrast weights mismatch for cope{k}')


def verify_design_con(path, spec):
    """Compare actual completed designs as well as current templates."""
    text = Path(path).read_text()
    if '/Matrix' not in text:
        raise PipelineError('Missing FEAT contrast matrix')
    matrix = np.loadtxt(text.split('/Matrix', 1)[1].splitlines(), ndmin=2)
    if matrix.shape[0] != spec['n_copes'] or not np.isfinite(matrix).all():
        raise PipelineError('Completed design contrast count/values invalid')
    for k, contract in spec['copes'].items():
        row = {i+1: float(v) for i, v in enumerate(matrix[k-1]) if v != 0}
        if row != contract['weights']:
            raise PipelineError('Completed design contrast weights disagree with contract')


def verify_l2(values, spec):
    expected = {'fmri(mixed_yn)': 3, 'fmri(npts)': 2, 'fmri(ncopeinputs)': spec['n_copes'],
                'fmri(evs_orig)': 1, 'fmri(ncon_real)': 1, 'fmri(evg1.1)': 1,
                'fmri(evg2.1)': 1, 'fmri(con_real1.1)': 1}
    try:
        valid = all(float(values.get(key, 'nan')) == value for key, value in expected.items())
    except ValueError:
        valid = False
    if not valid:
        raise PipelineError('L2 must be a two-run intercept-only fixed-effects mean with the expected cope count')


def repository_rows(c):
    rows = []
    for name, path in {'neural-signature': c.root, **c.repos}.items():
        def git(*args):
            return subprocess.check_output(['git', '--no-optional-locks', '-C', str(path), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        try:
            rows.append({'repository': 'sharedreward-aging' if name == 'aging' else 'rf1-sra-' + name, 'absolute_path': str(path),
                         'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
                         'head_sha': git('rev-parse', 'HEAD'),
                         'status': 'dirty' if git('status', '--porcelain') else 'clean'})
        except subprocess.CalledProcessError as exc:
            raise PipelineError(f'Cannot establish Git provenance for {name}') from exc
    return rows


def preflight(c):
    versions = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'nibabel', 'PyYAML', 'matplotlib']}
    for name, path in c.repos.items():
        print(f'  {name}: {path}', flush=True)
        if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
            raise PipelineError(f'Missing/unreadable source repository: {name}')
    for name, path in [('BIDS', c.bids), ('fMRIPrep', c.fmriprep), ('source exclusions', c.exclusions), ('TemplateFlow', c.templateflow), ('analysis checkout', c.root)]:
        print(f'  {name}: {path}', flush=True)
    if not c.bids.is_dir() or not (c.bids / 'participants.tsv').is_file():
        raise PipelineError('Canonical BIDS participants.tsv is required')
    if not c.exclusions.is_dir() or not os.access(c.exclusions, os.R_OK | os.X_OK):
        raise PipelineError('Authoritative source-exclusion directory must be readable; absence is not zero exclusions')
    for tree in ['work', 'results', 'reports', 'provenance']:
        parent = c.output(tree)
        while not parent.exists(): parent = parent.parent
        if not os.access(parent, os.W_OK | os.X_OK):
            raise PipelineError('Output directory is not writable')
    records = []
    for task, spec in c.contrasts.items():
        repo = c.repos[spec['repository']]
        if task == 'sharedreward':
            from aging_source import verify_contrast_table, load_aging_index
            path = verify_contrast_table(c)
            load_aging_index(c, refresh=True)
            records.append({'task': task, 'contrast_table': str(path.relative_to(repo)),
                            'sha256': sha256(path), **c.aging_provenance})
            print(f'  Shared Reward: verified aging full-trial audit; {len(c.aging_subjects)} RF1 subjects; '
                  'one-/two-run strategies retained; decision/neutral probes unavailable', flush=True)
            continue
        path = repo / spec['template']
        verify_l1(parse_fsf(path), spec)
        records.append({'task': task, 'template': str(path.relative_to(repo)), 'sha256': sha256(path)})
        if task not in ['socialdoors', 'doors']:
            l2 = parse_fsf(repo / f'templates/L2_task-{task}_model-{spec["model"]}_type-act.fsf')
            verify_l2(l2, spec)
        # Naming conventions are also tested against representatives in inventory.
        config_text = (repo / 'code/project_config.sh').read_text()
        token = 'smTo-' if task == 'sharedreward' else 'sm-'
        if token not in config_text:
            raise PipelineError('Upstream output naming changed; review paths')
    return repository_rows(c), records, versions
