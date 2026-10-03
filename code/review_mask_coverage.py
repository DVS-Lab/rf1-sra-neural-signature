"""Review an aggregate analysis mask against its verified public template; no participant images."""
from pathlib import Path
import argparse
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir()+"/rf1-neural-matplotlib")
import json, hashlib
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_config, sha256, PipelineError, write_json, atomic_output
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-dir', type=Path, required=True,
                    help='Contains template_brain_mask.nii.gz and template_T1w.nii.gz')
parser.add_argument('--regional-mask', type=Path, required=True,
                    help='Historical source-cerebellum-brainstem_mask.nii.gz from linux2/qc/reference')
args=parser.parse_args()
c=load_config()
root=c.root
ref=args.reference_dir
record=json.loads((root/'provenance/analysis_mask.json').read_text())
if sha256(ref/'template_brain_mask.nii.gz') != record['template_sha256']:
    raise PipelineError('Template mask hash differs from the completed run')
if sha256(root/'results/maps/analysis_mask.nii.gz') != record['sha256']:
    raise PipelineError('Analysis mask hash differs from the completed run')
if sha256(args.regional_mask) != '1335b40c2ad94056cd54c1b41aea100f5063428045c63271f7909432f4e310ed':
    raise PipelineError('Historical regional mask hash mismatch')
mask=nib.load(root/'results/maps/analysis_mask.nii.gz')
brain=nib.load(ref/'template_brain_mask.nii.gz')
t1=nib.load(ref/'template_T1w.nii.gz')
exclusion=nib.load(args.regional_mask)
b=brain.get_fdata()>0
m=resample_from_to(mask,brain,order=0).get_fdata()>0
fov_img=nib.Nifti1Image(np.ones(mask.shape,dtype=np.uint8),mask.affine)
fov=resample_from_to(fov_img,brain,order=0).get_fdata()>0
ex=resample_from_to(exclusion,brain,order=0).get_fdata()>0
native=mask.get_fdata()>0
brain_native=resample_from_to(brain,mask,order=0).get_fdata()>0
summary={'analysis_voxels':int(native.sum()),'analysis_voxel_mm':list(map(float,mask.header.get_zooms())),
         'native_grid_template_brain_voxels':int(brain_native.sum()),
         'native_grid_template_retained_pct':100*int((native&brain_native).sum())/int(brain_native.sum()),
         'template_sha256':hashlib.sha256((ref/'template_brain_mask.nii.gz').read_bytes()).hexdigest(),
         'mask_sha256':hashlib.sha256((root/'results/maps/analysis_mask.nii.gz').read_bytes()).hexdigest(),
         'comparison_grid':'Full TemplateFlow res-02 brain grid; nearest-neighbor mask resampling',
         'regional_reference':'Historical combined cerebellum/brainstem exclusion mask; not a fine anatomical atlas'}
for name,target in [('whole_brain',b),('cerebellum_brainstem',b&ex),('other_brain',b&~ex)]:
    n=int(target.sum()); kept=int((target&m).sum())
    summary[name]={'target_voxels':n,'retained_voxels':kept,'retained_pct':100*kept/n,
                   'outside_analysis_fov_pct':100*int((target&~fov).sum())/n}
missing=b&~m
coords=np.argwhere(missing)
world=nib.affines.apply_affine(brain.affine,coords)
summary['missing_voxel_world_bounds_mm']=[world.min(0).tolist(),world.max(0).tolist()]
write_json(c, 'reports/course_correction_review/mask_coverage.json', summary)
print(json.dumps(summary,indent=2))
a=resample_from_to(t1, brain, order=1).get_fdata()
fig,axes=plt.subplots(2,5,figsize=(15,7.2),facecolor='white')
for ax,z in zip(axes.ravel(),[-54,-42,-30,-18,-6,6,18,30,42,54]):
    k=int(round(nib.affines.apply_affine(np.linalg.inv(brain.affine),[0,0,z])[2]))
    ax.imshow(a[:,:,k].T,origin='lower',cmap='gray',vmin=0,vmax=np.percentile(a[b],99))
    layer=np.zeros(b[:,:,k].shape+(4,))
    layer[(b&m)[:,:,k]]=[0.0,0.75,0.85,0.35]
    layer[missing[:,:,k]]=[1.0,0.1,0.15,0.9]
    ax.imshow(layer.transpose(1,0,2),origin='lower')
    ax.set_title(f'MNI z = {z} mm'); ax.axis('off')
fig.suptitle('Actual analysis-mask coverage on the full MNI template\nCyan: retained brain | Red: omitted brain | Comparison on the 2-mm template grid',fontsize=14)
fig.tight_layout(rect=(0,0,1,.92))
with atomic_output(c, 'reports/course_correction_review/mask_coverage.png') as out:
    fig.savefig(out,dpi=160)
plt.close(fig)
