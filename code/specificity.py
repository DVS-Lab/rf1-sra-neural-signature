"""Exactly two development specificity analyses; production entry point."""
import argparse,json,importlib.metadata
import numpy as np
from pathlib import Path
from utils import load_config,PipelineError,sha256,write_json,write_tsv,atomic_output
from pilot import run_lock
from characterization_audit import audit,require,digest,PRIMARY
from characterization_compute import assert_development
from cross_valence import settings
from specificity_design import output_config,MODES,cv
from specificity_inputs import inventory,features,atlas_mask,verify_previous,overlap


def snapshot(base):
    result={}
    for tree in ('results','reports','provenance'):
        for p in sorted((base.root/tree).rglob('*')):
            rel=p.relative_to(base.root)
            if p.is_file() and tuple(rel.parts[1:3])!=('revised','specificity'): result[str(rel)]=sha256(p)
    for rel in ('work/splits/subject_split_v1.tsv','work/splits/development_folds.tsv','work/revised/samples/manifest.tsv'):
        p=base.root/rel
        if p.is_file(): result[rel]=sha256(p)
    return result


def checkpoint(out,key,mode,compute):
    rel='work/observed/'+mode; marker=out.output(rel+'.json'); path=out.output(rel+'.npy')
    if marker.exists():
        meta=json.loads(marker.read_text()); require(meta['fingerprint']==key and sha256(path)==meta['sha256'],'observed checkpoint drift')
        return np.load(path)
    result=compute()
    with atomic_output(out,rel+'.npy') as dest: np.save(dest,result)
    write_json(out,rel+'.json',dict(fingerprint=key,sha256=sha256(path)))
    return result


def run(base,workers=96,dry_run=False):
    require(workers>0,'workers must be positive'); out=output_config(base); before=None
    with run_lock(base):
        try:
            print('Specificity: auditing frozen development inputs and outputs',flush=True)
            spec=json.loads((base.root/'config/specificity.json').read_text())
            require(spec['n']==178 and spec['permutations']==500 and spec['bootstrap_samples']==10000,'fixed specificity design changed')
            for rel,h in spec['frozen'].items(): require(sha256(base.root/rel)==h,'frozen specificity dependency changed: '+rel)
            scope=audit(base,settings(base))[(PRIMARY,'three_paradigm')]
            require(len(scope['subjects'])==178,'retain exact original N=178; no sample reconstruction')
            assert_development(scope); before=snapshot(base)
            # All phase paths/QC are validated before any participant voxel read.
            phase_paths=inventory(scope)
            old_root=base.root/'work/revised/cross_valence'
            private={name:sha256(old_root/name) for name in ('features/three_paradigm/complete.json','oof_predictions.tsv')}
            source_hashes={str(p):sha256(p) for maps in phase_paths.values() for p in maps.values()}
            # These are allowed development source files only; no participant-directory sweep.
            assert_development(scope)
            software={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','nibabel','pandas')}
            identity=dict(spec=spec,original=before,private=private,phase_sources=source_hashes,
                subjects=scope['subjects'],folds=scope['folds'],qc=sha256(base.repos['linux2']/base.paths['qc_table']),software=software,
                code={p.name:sha256(p) for p in (base.root/'code').glob('specificity*.py')})
            key=digest(identity)
            if dry_run:
                print('DRY RUN PASSED: exact N=178, original folds/QC/mask and phase provenance checked. Development source bytes hashed; no voxel arrays loaded. No holdout images accessed.',flush=True)
                return
            marker=out.output('work/identity.json')
            if marker.exists(): require(json.loads(marker.read_text())['fingerprint']==key,'specificity inputs changed; retain old checkpoints for review')
            write_json(out,'work/identity.json',dict(fingerprint=key,**identity))
            for name in ('aggregate','figures','maps'): out.output('results/'+name).mkdir(parents=True,exist_ok=True)
            write_json(out,'provenance/run_status.json',dict(status='in_progress',fingerprint=key,holdout_scored=False))
            arrays,mask,ref=features(out,scope,key,phase_paths); keep,atlas=atlas_mask(base,mask,ref,spec)
            from build_mask import save_image
            save_image(out,'results/maps/fixed_dmn_intersection.nii.gz',(atlas&mask).astype('uint8'),ref)
            info=dict(mask_voxels=int(mask.sum()),dmn_voxels=int(keep.sum()),excluded_voxels=int((~keep).sum()),atlas_sha256=spec['atlas_sha256'],mask_sha256=scope['model']['mask_sha256'])
            write_json(out,'provenance/reference.json',info)
            membership=[dict(subject=s,fold=f,role='test' if scope['folds'][s]==f else 'train') for f in range(1,6) for s in scope['subjects']]
            write_tsv(out,'work/model_membership.tsv',membership)
            observed={}
            for mode in MODES:
                observed[mode]=checkpoint(out,key,mode,lambda mode=mode:cv(scope,arrays,mode,keep))
                if mode=='legacy': write_tsv(out,'results/aggregate/old_prediction_reconstruction.tsv',verify_previous(scope,observed[mode]))
                if mode=='outcome':
                    require(np.allclose(observed[mode][:,2:6,2:6],observed['legacy'][:,2:6,2:6],atol=1e-5,rtol=2e-5),'unchanged Trust/Doors predictions differ')
                print('Finished specificity observed '+mode,flush=True)
            from specificity_parallel import run as permutations
            null=permutations(out,scope,arrays,keep,key,requested=workers,count=spec['permutations'])
            from specificity_report import report
            frame=report(out,observed,null,overlap(base,scope,spec),info)
            require(snapshot(base)==before,'pre-existing output changed')
            write_json(out,'provenance/run_status.json',dict(status='complete',fingerprint=key,n=178,holdout_scored=False,
                original_outputs_unchanged=True,permutations_per_variant=500,null_center_flags=int(frame.null_center_flag.sum()),
                no_ugr_or_betrayal=True,no_candidate_changes=True))
            print(out.output('reports/SPECIFICITY.md'),flush=True)
        except Exception as exc:
            if before is not None:
                write_json(out,'provenance/run_status.json',dict(status='failed',error_type=type(exc).__name__,holdout_scored=False,original_outputs_unchanged=snapshot(base)==before))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--workers',type=int,default=96); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
    try: run(load_config(),a.workers,a.dry_run)
    except (PipelineError,FileNotFoundError) as exc: print('ERROR: '+str(exc)); return 1
    return 0
if __name__=='__main__': raise SystemExit(main())
