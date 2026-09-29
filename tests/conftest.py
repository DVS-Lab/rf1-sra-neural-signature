import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import json
import shutil
import subprocess
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml
from utils import load_config
from inventory import feat_dir


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    root = tmp_path / 'analysis'
    root.mkdir()
    shutil.copytree(Path(__file__).resolve().parents[1] / 'config', root / 'config')
    paths = yaml.safe_load((root / 'config/paths.yaml').read_text())
    paths['projects_root'] = str(tmp_path / 'sources')
    paths['source_exclusions'] = str(tmp_path / 'exclusions')
    paths['templateflow'] = str(tmp_path / 'templateflow')
    (root / 'config/paths.yaml').write_text(yaml.safe_dump(paths))
    for key in ['PROJECTS_ROOT', 'BIDS_ROOT', 'FMRIPREP_ROOT', 'SOURCEDATA_EXCLUSIONS_ROOT', 'TEMPLATEFLOW_HOME']:
        monkeypatch.delenv(key, raising=False)
    for key in paths['repositories']: monkeypatch.delenv('RF1_'+key.upper()+'_ROOT', raising=False)
    analysis_path = root / 'config/analysis.yaml'
    analysis = yaml.safe_load(analysis_path.read_text())
    analysis['minimum_multitask_n'] = 55  # Explicit synthetic-only small fixture, never production default.
    analysis['bootstrap_samples'] = 300
    analysis_path.write_text(yaml.safe_dump(analysis))
    c = load_config(root=root)
    for repo in [root, *c.repos.values()]:
        repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '--allow-empty', '-qm', 'synthetic fixture'], check=True)
    c.bids.mkdir()
    c.exclusions.mkdir()
    for task, spec in c.contrasts.items():
        repo = c.repos[spec['repository']]
        template = repo / spec['template']
        template.parent.mkdir(exist_ok=True)
        template.write_text(fsf_text(spec, 'DATA'))
        if task not in ['socialdoors', 'doors']:
            (template.parent / f'L2_task-{task}_model-{spec["model"]}_type-act.fsf').write_text(
                l2_text(spec))
        (repo / 'code').mkdir(exist_ok=True)
        (repo / 'code/project_config.sh').write_text('smTo-6 sm-5\n')
    return c


def l2_text(spec):
    return (f'set fmri(mixed_yn) 3\nset fmri(npts) 2\nset fmri(ncopeinputs) {spec["n_copes"]}\n'
            'set fmri(evs_orig) 1\nset fmri(ncon_real) 1\nset fmri(evg1.1) 1\n'
            'set fmri(evg2.1) 1\nset fmri(con_real1.1) 1\n')


def fsf_text(spec, data, rendered=False):
    smooth = '5' if rendered and spec['smoothing'] == 'SMOOTH' else spec['smoothing']
    lines = [f'set fmri(ncon_real) {spec["n_copes"]}', f'set fmri(smooth) {smooth}',
             'set fmri(regstandard_yn) 0', f'set feat_files(1) "{data}"']
    lines += [f'set fmri(evtitle{i}) \"{title}\"' for i, title in spec['ev_titles'].items()]
    for k, value in spec['copes'].items():
        lines.append(f'set fmri(conname_real.{k}) "{value["name"]}"')
        lines += [f'set fmri(con_real{k}.{i}) {weight}' for i, weight in value['weights'].items()]
    return '\n'.join(lines)+'\n'


def con_text(spec):
    columns = max(i for cope in spec['copes'].values() for i in cope['weights'])
    matrix = np.zeros((spec['n_copes'], columns))
    for k, value in spec['copes'].items():
        for i, weight in value['weights'].items(): matrix[k-1, i-1] = weight
    return '/Matrix\n' + '\n'.join(' '.join(str(v) for v in row) for row in matrix) + '\n'


def save_nii(path, data, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([2., 2., 2., 1.]) if affine is None else affine
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    img.header.set_xyzt_units('mm', 'sec')
    img.set_qform(affine, 4); img.set_sform(affine, 4)
    nib.save(img, path)


def make_subject(c, subject, secondary=True, index=0):
    rng = np.random.default_rng(index)
    shape = (3, 4, 5)
    spatial = np.linspace(-1, 1, np.prod(shape)).reshape(shape)
    tasks = list(c.contrasts) if secondary else ['sharedreward']
    for task in tasks:
        spec = c.contrasts[task]
        runs = [1] if task in ['doors', 'socialdoors'] else [1, 2]
        for run in runs:
            func = c.bids / subject / 'ses-01/func'
            bold = c.fmriprep / subject / 'ses-01/func' / f'{subject}_ses-01_task-{task}_run-{run}_part-mag_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
            save_nii(bold, np.ones(shape + (2,)))
            if task == 'sharedreward':
                raw = func / f'{subject}_ses-01_task-{task}_run-{run}_echo-1_part-mag_bold.nii.gz'
                save_nii(raw, np.ones(shape + (2,)))
                raw.with_name(raw.name.replace('.nii.gz', '.json')).write_text(json.dumps({'FlipAngle': 20 if index % 2 else 50}))
            base = feat_dir(c, subject, task, 'L1', run)
            base.mkdir(parents=True, exist_ok=True)
            (base / 'design.fsf').write_text(fsf_text(spec, bold, True))
            (base / 'design.con').write_text(con_text(spec))
            (base / 'design.mat').write_text('fixture')
            save_nii(base / 'mask.nii.gz', np.ones(shape))
            save_nii(base / 'cluster_mask_zstat1.nii.gz', np.ones(shape))
            for k in range(1, spec['n_copes'] + 1):
                direction = (1 if k in [4, 6, 7, 9, 14] else -.2)
                if task == 'doors': direction = -.7
                data = direction * spatial + rng.normal(0, .12, shape)
                save_nii(base / f'stats/cope{k}.nii.gz', data)
        if task in ['doors', 'socialdoors']: continue
        base = feat_dir(c, subject, task)
        base.mkdir(parents=True, exist_ok=True)
        (base / 'design.fsf').write_text(l2_text(spec) + '\n'.join(
            f'set feat_files({r}) "{feat_dir(c,subject,task,"L1",r)}"' for r in [1, 2]))
        for item in ['design.mat', 'design.con']: (base / item).write_text('fixture')
        for k in range(1, spec['n_copes'] + 1):
            cope = base / f'cope{k}.feat'
            cope.mkdir()
            for item in ['design.mat', 'design.con']: (cope / item).write_text('fixture')
            save_nii(cope / 'mask.nii.gz', np.ones(shape))
            save_nii(cope / 'cluster_mask_zstat1.nii.gz', np.ones(shape))
            save_nii(cope / 'stats/zstat1.nii.gz', np.ones(shape))
            a = nib.load(feat_dir(c, subject, task, 'L1', 1) / f'stats/cope{k}.nii.gz').get_fdata()
            b = nib.load(feat_dir(c, subject, task, 'L1', 2) / f'stats/cope{k}.nii.gz').get_fdata()
            save_nii(cope / 'stats/cope1.nii.gz', (a+b)/2)


def metadata(n=100):
    return pd.DataFrame({'subject': [f'sub-fixture{i:03}' for i in range(n)],
                         'age': np.linspace(20, 85, n), 'sex': ['F' if i%2 else 'M' for i in range(n)],
                         'flip_angle': ['20' if i%2 else '50' for i in range(n)],
                         **{task: [True]*n for task in ['sharedreward','trust','socialdoors','doors','ugr']}})


@pytest.fixture
def cohort(cfg):
    data = metadata(70)
    for i, row in data.iterrows(): make_subject(cfg, row.subject, secondary=True, index=i)
    data.rename(columns={'subject': 'participant_id'}).to_csv(cfg.bids / 'participants.tsv', sep='\t', index=False)
    return cfg, data
