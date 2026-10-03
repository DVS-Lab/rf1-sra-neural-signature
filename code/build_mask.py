"""Development-only voxel access, fixed mask, and consistent vector ordering."""
import json
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from inventory import cope_path, feat_dir, mask_paths, header_info, same_grid, InputUnavailable
from utils import PipelineError, atomic_output, sha256, write_json
from aging_source import subject_unit


def load_development_image(c, path, subject, guard, development_subjects):
    guard.check([subject], development_subjects)  # Must precede nib.load/get_fdata.
    path = Path(path)
    # Prevent assigning a development ID to another participant's source image.
    if subject not in path.parts or 'ses-01' not in path.parts:
        raise PipelineError('Image path does not belong to authorized participant/session')
    if not any(path.resolve().is_relative_to(repo.resolve()) for repo in c.repos.values()):
        raise PipelineError('Image lies outside configured sources')
    if subject not in path.resolve().parts:
        raise PipelineError('Resolved image lacks the authorized participant identity')
    for component in path.resolve().parts:
        if component.startswith('sub-') and component != subject:
            raise PipelineError('Image symlink resolves to another participant')
    header_info(path, c.analysis['space'])
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all(): raise InputUnavailable('nonfinite_voxels')
    return nib.Nifti1Image(data, img.affine, img.header)


def save_image(c, relative, data, reference):
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    # Public maps must not inherit participant-bearing free text/extensions.
    header['descrip'] = b'RF1-SRA development-derived map'
    header['aux_file'] = b''
    header.set_intent('none', name='')
    header.extensions.clear()
    image = nib.Nifti1Image(data, reference.affine, header)
    with atomic_output(c, relative) as path: nib.save(image, path)


def build_mask(c, guard, development_subjects):
    subjects = sorted(development_subjects)
    guard.check(subjects, development_subjects)
    reference_path = cope_path(c, subjects[0], 'sharedreward', 1)
    # A development participant supplies geometry, without condition-dependent masking.
    reference = nib.load(reference_path)
    counts = np.zeros(reference.shape, dtype=np.uint32)
    mask_sources = []
    for subject in subjects:
        # Common spatial support across all three clean paradigm pairs, with no labels.
        participant = np.ones(reference.shape, dtype=bool)
        for task in ['sharedreward', 'trust', 'socialdoors', 'doors']:
            paths = mask_paths(c, subject, task)
            for path in paths:
                image = load_development_image(c, path, subject, guard, development_subjects)
                resampled = not same_grid(image, reference)
                if resampled:
                    if task == 'sharedreward':
                        raise PipelineError('Shared Reward reference mask grids disagree')
                    image = resample_from_to(image, (reference.shape, reference.affine), order=0)
                participant &= image.get_fdata(dtype=np.float32) > 0
                mask_sources.append({'subject': subject, 'task': task, 'path': str(path),
                                     'resampled': resampled, 'interpolation': 'nearest' if resampled else 'none'})
        counts += participant
    from utils import write_tsv
    write_tsv(c, 'work/diagnostics/mask_sources.tsv', mask_sources)
    mask = counts / len(subjects) >= c.analysis['coverage']
    template_candidates = sorted(c.templateflow.glob('tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz'))
    template = None
    method = 'development_multitask_FEAT_coverage_only; TemplateFlow mask unavailable'
    if len(template_candidates) == 1:
        template = template_candidates[0]
        header_info(template, c.analysis['space'])
        img = nib.load(template)
        img = resample_from_to(img, (reference.shape, reference.affine), order=0)
        data = img.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all(): raise PipelineError('TemplateFlow mask contains nonfinite values')
        mask &= data > 0
        method = 'TemplateFlow_MNI152NLin6Asym_intersect_95pct_development_multitask_FEAT'
    if mask.sum() < 2: raise PipelineError('Analysis mask has fewer than two voxels')
    save_image(c, 'results/maps/analysis_mask.nii.gz', mask.astype(np.uint8), reference)
    # Keep participant-bearing reference paths private.
    write_json(c, 'work/diagnostics/mask_reference.json', {'reference_image': str(reference_path)})
    write_json(c, 'provenance/analysis_mask.json',
               {'method': method, 'coverage_threshold': c.analysis['coverage'], 'development_n': len(subjects),
                'coverage_tasks': ['sharedreward', 'trust', 'socialdoors', 'doors'],
                'ugr_masks_used': False,
                'voxel_count': int(mask.sum()), 'dimensions': reference.shape,
                'voxel_sizes': reference.header.get_zooms(), 'affine': reference.affine,
                'reference': 'First sorted development Shared Reward verified subject-output grid; exact path in local work/diagnostics/mask_reference.json',
                'template': str(template) if template else None,
                'template_sha256': sha256(template) if template else None,
                'sha256': sha256(c.output('results/maps/analysis_mask.nii.gz'))})
    print(f'  Fixed mask: {mask.sum()} voxels; {method}', flush=True)
    return mask, reference


def spatial_center(data, mask):
    vector = np.asarray(data[mask], dtype=np.float32)
    if not np.isfinite(vector).all(): raise InputUnavailable('nonfinite_vector')
    return vector - np.float32(vector.mean(dtype=np.float64))


def reconstruct(vector, mask):
    if len(vector) != int(mask.sum()): raise PipelineError('Coefficient/mask length mismatch')
    data = np.zeros(mask.shape, dtype=np.float32)
    data[mask] = vector
    return data


def vectorize_source(c, subject, task, k, mask, reference, guard, development_subjects,
                     diagnostics, level='L2', run=None):
    if task == 'sharedreward' and level == 'L2': level, run = subject_unit(c, subject)
    path = cope_path(c, subject, task, k, level, run)
    image = load_development_image(c, path, subject, guard, development_subjects)
    resampled = not same_grid(image, reference)
    if resampled:
        if task == 'sharedreward': raise PipelineError('Primary grids differ; require upstream geometry review')
        # Space identity has already been established from actual L1 design provenance.
        image = resample_from_to(image, (reference.shape, reference.affine), order=1)
    data = image.get_fdata(dtype=np.float32)
    vector = data[mask].astype(np.float32)
    if not np.isfinite(vector).all(): raise InputUnavailable('nonfinite_resampled_voxels')
    diagnostics.append({'subject': subject, 'task': task, 'level': level, 'run': run, 'cope': k,
                        'path': str(path), 'resampled': resampled, 'interpolation': 'linear' if resampled else 'none',
                        'source_norm': float(np.linalg.norm(vector)), 'source_mean': float(vector.mean()),
                        'voxel_finite': True})
    return vector
