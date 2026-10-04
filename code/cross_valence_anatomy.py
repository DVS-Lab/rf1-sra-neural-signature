"""Local standard anatomy only; independent visual exclusion is proposed, never applied."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from characterization_audit import require, PRIMARY
from characterization_compute import assert_development, geometry
from utils import sha256, write_json, write_text, write_tsv

VISUAL_LABELS=('Intracalcarine Cortex','Supracalcarine Cortex','Cuneal Cortex','Lingual Gyrus',
               'Occipital Fusiform Gyrus','Occipital Pole','Lateral Occipital Cortex, inferior division',
               'Lateral Occipital Cortex, superior division')


def inventory(base):
    roots=[Path(p) for p in [os.environ.get('FSLDIR',''),'/usr/local/fsl','/opt/fsl','/ZPOOL/data/tools/fsl'] if p]
    tools=Path('/ZPOOL/data/tools')
    if tools.exists(): roots.extend(sorted(p for p in tools.glob('fsl*') if p.is_dir()))
    roots=list(dict.fromkeys(p.resolve() for p in roots if p.is_dir()))
    result={'fsl_roots':[],'available_atlas_definitions':[],'background_candidates':[],'selected_atlases':[]}
    for root in roots:
        version=root/'etc/fslversion'
        ver=version.read_text().strip() if version.is_file() else 'version file unavailable; SHA256 identifies selected assets'
        result['fsl_roots'].append({'path':str(root),'version':ver})
        for p in sorted((root/'data/atlases').glob('*.xml')):
            result['available_atlas_definitions'].append({'path':str(p),'name':p.stem})
        bg=root/'data/standard/MNI152_T1_2mm_brain.nii.gz'
        if bg.is_file(): result['background_candidates'].append(str(bg))
        for name,xml in [('cort','HarvardOxford-Cortical.xml'),('sub','HarvardOxford-Subcortical.xml')]:
            image=root/f'data/atlases/HarvardOxford/HarvardOxford-{name}-maxprob-thr25-2mm.nii.gz'
            label=root/'data/atlases'/xml
            if image.is_file() and label.is_file() and not any(a['kind']==name for a in result['selected_atlases']):
                result['selected_atlases'].append(dict(kind=name,name='Harvard-Oxford '+name+' maxprob thr25 2mm',
                    image=str(image),labels=str(label),fsl_version=ver,image_sha256=sha256(image),labels_sha256=sha256(label)))
    tf=base.templateflow/'tpl-MNI152NLin6Asym'
    if tf.exists():
        for p in sorted(tf.glob('*res-02*desc-brain_T1w.nii.gz')): result['background_candidates'].append(str(p))
        for p in sorted(tf.glob('*res-02_T1w.nii.gz')): result['background_candidates'].append(str(p))
        result['available_atlas_definitions'].extend({'path':str(p),'name':p.name} for p in sorted(tf.glob('*atlas*.tsv')))
    # Prefer the exact configured template space, otherwise local FSL MNI152 background for display only.
    candidates=result['background_candidates']
    candidates.sort(key=lambda p:('tpl-MNI152NLin6Asym' not in p,p))
    result['background']=candidates[0] if candidates else None
    if result['background']: result['background_sha256']=sha256(result['background'])
    result['visual_proposal_selection_rule']='Harvard-Oxford cortical maxprob thr25 2mm, fixed eight anatomical occipital labels; no result-dependent fallback'
    return result


def standard_image(scope,path,expected):
    assert_development(scope)
    path=Path(path)
    require(not any(p.startswith('sub-') for p in path.parts),'participant image cannot be standard anatomy')
    require(sha256(path)==expected,'standard anatomy asset changed')
    return nib.load(path)


def label_names(asset):
    require(sha256(asset['labels'])==asset['labels_sha256'],'atlas label XML changed')
    return {int(e.attrib['index'])+1:(e.text or '').strip() for e in ET.parse(asset['labels']).getroot().findall('.//data/label')}


def prepare(out,base,scopes,assets):
    """Append labels to OLD descriptive peaks and count a proposed independent mask only."""
    scope=scopes[(PRIMARY,'partner_pair')]; assert_development(scope)
    write_json(out,'provenance/local_anatomy.json',assets)
    loaded=[]
    for asset in assets['selected_atlases']:
        image=standard_image(scope,asset['image'],asset['image_sha256'])
        loaded.append((asset,image,image.get_fdata(),label_names(asset)))
    source=base.root/'results/revised/characterization/aggregate/descriptive_clusters.tsv'
    if source.exists():
        peaks=pd.read_csv(source,sep='\t'); rows=[]
        for row in peaks.to_dict('records'):
            labels=[]
            for asset,img,data,names in loaded:
                ijk=np.rint(nib.affines.apply_affine(np.linalg.inv(img.affine),[row['x'],row['y'],row['z']])).astype(int)
                value=int(data[tuple(ijk)]) if np.all(ijk>=0) and np.all(ijk<np.array(data.shape)) else 0
                if value in names: labels.append(asset['name']+': '+names[value])
            rows.append({**row,'descriptive_peak_label':'; '.join(labels) or 'unlabeled / local atlas unavailable',
                         'source':'completed characterization bootstrap-stable peaks; no new spatial bootstrap'})
        write_tsv(out,'results/aggregate/descriptive_peak_labels.tsv',rows)
    body='# Visual-region sensitivity proposal — NOT EXECUTED\n\n'
    body+='This independent definition was fixed in code before new results. It is a proposal only: no source features are excluded and no classifier is fit with this mask. Prospective review and approval are required.\n\n'
    cortical=next((x for x in loaded if x[0]['kind']=='cort'),None)
    proposal={'status':'unavailable','executed':False,'labels':list(VISUAL_LABELS)}
    if cortical:
        asset,img,data,names=cortical; missing=set(VISUAL_LABELS)-set(names.values())
        require(not missing,'installed cortical atlas lacks prespecified visual labels: '+str(sorted(missing)))
        ids=[k for k,v in names.items() if v in VISUAL_LABELS]
        binary=np.isin(data,ids); counts={}
        for cohort in ('partner_pair','three_paradigm'):
            sc=scopes[(PRIMARY,cohort)]; assert_development(sc); mask,ref=geometry(sc)
            resampled=resample_from_to(nib.Nifti1Image(binary.astype(np.uint8),img.affine),(mask.shape,ref.affine),order=0).get_fdata()>0
            counts[cohort]={'existing_mask_voxels':int(mask.sum()),'proposed_excluded_voxels':int((mask&resampled).sum()),
                            'proposed_remaining_voxels':int((mask&~resampled).sum())}
        proposal.update(status='prepared_for_review',atlas=asset,label_values=ids,native_voxels=int(binary.sum()),counts=counts)
        body+=f"Source: {asset['name']}; installed FSL version: {asset['fsl_version']}.\n\nImage: `{asset['image']}`\n\nImage SHA256: `{asset['image_sha256']}`\n\nLabel XML: `{asset['labels']}`\n\nXML SHA256: `{asset['labels_sha256']}`\n\n"
        body+='Exact integer values and labels (XML zero-based indices +1 in maxprob image):\n\n'
        body+='\n'.join(f'- {i}: {names[i]}' for i in ids)+'\n\n'
        body+='Proposed rule: nearest-neighbor resample the union of these labels to each frozen analysis grid, then exclude its intersection with that original analysis mask. Recenter each remaining map independently. Preserve participants, folds, contrasts and classifier settings. Do not optimize the threshold or parcel set.\n\n'
        body+=f'Native union: {int(binary.sum())} voxels.\n\n'
        for c,vals in counts.items(): body+=f'- {c}: {vals["proposed_excluded_voxels"]} of {vals["existing_mask_voxels"]} voxels excluded; {vals["proposed_remaining_voxels"]} retained.\n'
        body+='\nRationale: a fixed anatomical occipital definition spanning medial and lateral occipital cortex, lingual and occipital fusiform tissue. This is not a complete functional visual-network mask; temporal face-sensitive tissue and perceptual information elsewhere may remain. Conversely, included association cortex need not be exclusively visual. Persistence would not prove abstract social coding.\n'
    else:
        body+='The prespecified Harvard-Oxford cortical asset was not found locally. No alternative was selected and nothing was downloaded. Exact labels are listed below, but voxel counts and asset identity remain unresolved until an existing installation is located.\n\n'
        body+='\n'.join('- '+label for label in VISUAL_LABELS)+'\n'
    body+='\nInventory and exact asset fingerprints: `provenance/revised/cross_valence/local_anatomy.json`. Peak labels describe individual coordinates, not the identity of a whole distributed component. No atlas significance or regional necessity is inferred.\n\nholdout_scored=False.\n'
    write_json(out,'provenance/visual_proposal.json',proposal)
    write_text(out,'reports/VISUAL_SENSITIVITY_PROPOSAL.md',body)
    return proposal
