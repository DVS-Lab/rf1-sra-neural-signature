"""Final, bounded empirical task-deactivation control; development only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np
from utils import load_config, PipelineError, sha256, write_json, write_text
from pilot import run_lock
from characterization_audit import audit, require, digest, PRIMARY, PARAMETERS, SEED
from characterization_compute import assert_development, geometry
from cross_valence import settings
from task_negative_design import output_config, REFERENCE
from task_negative_sources import prior_inputs, audit_sources, raw_features


def snapshot(base):
    result={}
    for tree in ('results','reports','provenance'):
        for p in sorted((base.root/tree).rglob('*')):
            rel=p.relative_to(base.root)
            if p.is_file() and tuple(rel.parts[1:3])!=('revised','task_negative_control'):
                result[str(rel)]=sha256(p)
    for rel in ('work/splits/subject_split_v1.tsv','work/splits/development_folds.tsv','work/revised/samples/manifest.tsv'):
        if (base.root/rel).is_file(): result[rel]=sha256(base.root/rel)
    return result


def background(base):
    # Standard anatomy for display only: no new atlas or source-image search.
    previous=base.root/'provenance/revised/cross_valence/local_anatomy.json'
    recorded=json.loads(previous.read_text())['background']
    candidates=[Path(recorded)] if recorded else []
    if os.environ.get('FSLDIR'): candidates.append(Path(os.environ['FSLDIR'])/'data/standard/MNI152_T1_2mm_brain.nii.gz')
    for p in candidates:
        if p.is_file():
            require(not any(x.startswith('sub-') for x in p.resolve().parts),'participant image cannot be anatomical background')
            return dict(path=str(p),sha256=sha256(p))
    raise PipelineError('Existing MNI anatomical background unavailable; set FSLDIR to installed FSL. No download or alternative atlas.')


def identity_check(out,identity,write=False):
    key=digest(identity); p=out.output('work/identity.json')
    if p.exists(): require(json.loads(p.read_text())==dict(identity,fingerprint=key),'task-negative input/code/software drift; preserve checkpoints and investigate')
    if write: write_json(out,'work/identity.json',dict(identity,fingerprint=key))
    return key


def run(base,workers=96,dry_run=False):
    require(workers>0,'workers must be positive'); out=output_config(base); before=None; stage='audit'
    with run_lock(base):
        try:
            spec=json.loads((base.root/'config/task_negative_control.json').read_text())
            require((spec['n'],spec['permutations'],spec['bootstrap_samples'],spec['reference'])==(178,500,10000,REFERENCE),'fixed empirical design changed')
            require(spec['classifier']==PARAMETERS and spec['seed']==SEED,'classifier/seed drift')
            for rel,h in spec['frozen'].items(): require(sha256(base.root/rel)==h,'pre-existing dependency changed: '+rel)
            before=snapshot(base)
            print('Task-negative control: exact frozen development cohort, outcome caches and baseline audit',flush=True)
            scope=audit(base,settings(base))[(PRIMARY,'three_paradigm')]
            require(len(scope['subjects'])==178,'exact N=178 required; no cohort reconstruction')
            assert_development(scope)
            previous,private=prior_inputs(base,scope)
            paths,sources=audit_sources(out,scope,spec)
            bg=background(base)
            software={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','nibabel','pandas','matplotlib','nilearn')}
            identity=dict(spec=spec,previous=before,private=private,sources=sources,subjects=scope['subjects'],folds=scope['folds'],
                mask_sha256=scope['model']['mask_sha256'],qc=sha256(base.repos['linux2']/base.paths['qc_table']),background=bg,
                software=software,code={p.name:sha256(p) for p in (base.root/'code').glob('task_negative*.py')})
            key=identity_check(out,identity,write=not dry_run)
            if dry_run:
                require(snapshot(base)==before,'pre-existing output changed during audit')
                print('DRY RUN PASSED: actual contrast weights, canonical event timings, interior fixation, QC, frozen membership/cache hashes. Source bytes hashed and headers inspected; no participant voxel arrays loaded. Template plausibility and new predictions remain untested. Holdout not accessed.',flush=True)
                return
            def status(stage,state='in_progress',**extra):
                write_json(out,'provenance/run_status.json',dict(status=state,stage=stage,fingerprint=key,n=178,holdout_scored=False,**extra))
            stage='signed decision sources'; status(stage)
            for name in ('aggregate','maps','figures'): out.output('results/'+name).mkdir(parents=True,exist_ok=True)
            mask,ref=geometry(scope); x=np.load(previous['outcome'],mmap_mode='r')
            require(x.shape==(178,10,2,int(mask.sum())),'frozen outcome cache geometry changed')
            # Inspect a participant at a time to avoid a second multi-GB full-array allocation.
            for z in x:
                require(np.isfinite(z).all() and np.allclose(z.mean(-1),0,atol=1e-4,rtol=0),'invalid or uncentered outcome features')
            raw=raw_features(out,scope,paths,mask,ref,key)
            from specificity_inputs import atlas_mask
            dmn,_=atlas_mask(base,mask,ref,spec)
            from task_negative_compute import templates,observed
            directions=templates(out,scope,raw,x,dmn,mask,ref,key)
            from task_negative_report import sanity_text,report
            sanity_body=sanity_text(out)
            write_text(out,'reports/REPORT.md','# Empirical task-negative control — IN PROGRESS\n\n'+sanity_body+'\n\nPredictive comparisons are pending. Validation remains unscored.\n')
            print(sanity_body,flush=True)
            stage='matched observed predictions'; status(stage)
            margins,expressions=observed(out,scope,x,directions,np.load(previous['observed']),key)
            stage='primary participant permutations'; status(stage)
            from task_negative_parallel import run as permutations
            null=permutations(out,scope,x,directions,key,requested=workers,count=spec['permutations'])
            stage='report and figure'; status(stage)
            frame=report(out,margins,expressions,null,bg)
            require(snapshot(base)==before,'pre-existing outputs changed')
            write_json(out,'provenance/analysis.json',dict(reference=REFERENCE,software=software,code=identity['code'],
                classifier=PARAMETERS,one_feature_solver='dual=False; same C, tolerance, loss and penalty',seed=SEED,
                mask_sha256=scope['model']['mask_sha256'],qc_sha256=identity['qc'],baseline_evidence=spec['baseline_evidence'],
                source_fingerprint=digest(sources),original_snapshot_fingerprint=digest(before),background=bg,
                permutations=500,bootstrap_samples=10000,holdout_scored=False))
            status('finished','complete',original_outputs_unchanged=True,no_candidate_changes=True,no_ugr_or_betrayal=True,
                   null_center_flags=int(frame[frame.role=='primary'].null_center_flag.sum()))
            print(out.output('reports/REPORT.md'),flush=True)
            print('STOP: final development control complete. No holdout scores, new candidates or further exploration.',flush=True)
        except Exception as exc:
            if before is not None:
                write_json(out,'provenance/run_status.json',dict(status='failed',stage=stage,error_type=type(exc).__name__,holdout_scored=False,original_outputs_unchanged=snapshot(base)==before))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--workers',type=int,default=96); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
    try: run(load_config(),a.workers,a.dry_run)
    except (PipelineError,FileNotFoundError) as exc: print('ERROR: '+str(exc)); return 1
    return 0
if __name__=='__main__': raise SystemExit(main())
