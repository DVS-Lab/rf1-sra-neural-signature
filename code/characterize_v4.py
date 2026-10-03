"""Characterize frozen completed v4 development models. No validation scoring option."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
import yaml
from pilot import run_lock
from utils import load_config, PipelineError, sha256, write_json, write_tsv, write_text, atomic_output
from characterization_audit import (audit, output_config, frozen_snapshot, digest, write_audit,
                                    require, PRIMARY, SENSITIVITY, TARGETS, TASKS)
from characterization_compute import (geometry, coefficients, save_vector, correlation, haufe, build_arrays,
                                      verify_oof, bootstrap, permutation, clusters, assert_development)
from build_mask import reconstruct


def settings(base):
    spec=yaml.safe_load((base.root/'config/characterization.yaml').read_text())
    require(spec['seed']==20260928 and spec['bootstrap_samples']==1000 and spec['permutations']==200,
            'production characterization requires the requested frozen resampling defaults')
    require(spec['sign_stability']==.975 and spec['minimum_cluster_voxels']==20, 'display rule changed')
    return spec


def intersection_correlation(a,mask_a,ref_a,b,mask_b,ref_b):
    same=mask_a.shape==mask_b.shape and np.allclose(ref_a.affine,ref_b.affine)
    va=reconstruct(a,mask_a); vb=reconstruct(b,mask_b)
    if not same:
        vb=resample_from_to(nib.Nifti1Image(vb,ref_b.affine),(mask_a.shape,ref_a.affine),order=1).get_fdata()
        mask_b=resample_from_to(nib.Nifti1Image(mask_b.astype(np.uint8),ref_b.affine),(mask_a.shape,ref_a.affine),order=0).get_fdata()>0
    valid=mask_a&mask_b&np.isfinite(va)&np.isfinite(vb)
    return correlation(va[valid],vb[valid]),int(valid.sum()),not same


def descriptive_maps(out,scope,cohort,arrays,mask,ref):
    patterns={}; rows=[]
    for family in TARGETS[cohort]:
        x=arrays[family]
        for name,tasks in [(t,(t,)) for t in TASKS[cohort]]+[('common',TASKS[cohort])]:
            model_name=family+'_'+('common' if name=='common' else 'task-'+name)
            entry=scope['models'][(model_name,0)]
            w,b=coefficients(scope,entry,mask,ref)
            idx=[TASKS[cohort].index(t) for t in tasks]; selected=x[:,idx]
            a=haufe(selected,w,b); mean=(selected[:,:,0]-selected[:,:,1]).mean(axis=(0,1),dtype=np.float64)
            stem=cohort+'_'+family+'_'+name
            # Byte-for-byte copy: raw coefficient NIfTI is not normalized or rewritten.
            with atomic_output(out,'results/maps/'+stem+'_raw_weights.nii.gz') as dest:
                shutil.copyfile(scope['c'].root/entry['maps'][0]['path'],dest)
            require(sha256(out.output('results/maps/'+stem+'_raw_weights.nii.gz'))==entry['maps'][0]['sha256'],'raw-copy mismatch')
            save_vector(out,stem+'_haufe_pattern',a,mask,ref); save_vector(out,stem+'_mean_difference',mean,mask,ref)
            patterns[(cohort,family,name)]={'weight':w,'haufe':a,'mean_difference':mean}
            rows.append({'cohort':cohort,'family':family,'model':name,'n_participants':len(x),'n_training_maps':int(np.prod(selected.shape[:3])),
                         'weight_haufe_r':correlation(w,a),'weight_mean_r':correlation(w,mean),'haufe_mean_r':correlation(a,mean)})
    return patterns,rows


def spatial_comparisons(out,scopes,patterns,geometries):
    robust=[]; cross=[]
    for cohort,families in TARGETS.items():
        primary=scopes[(PRIMARY,cohort)]; sensitivity=scopes[(SENSITIVITY,cohort)]
        mask,ref=geometries[cohort]; smask,sref=geometry(sensitivity)
        for family in families:
            w,_=coefficients(sensitivity,sensitivity['models'][(family+'_common',0)],smask,sref)
            r,n,resampled=intersection_correlation(patterns[(cohort,family,'common')]['weight'],mask,ref,w,smask,sref)
            p=primary['performance'].query('family == @family and scope == "common"').set_index('test_task')
            s=sensitivity['performance'].query('family == @family and scope == "common"').set_index('test_task')
            for task in TASKS[cohort]:
                robust.append({'cohort':cohort,'family':family,'task':task,'primary_n':int(p.loc[task,'n']),
                               'sensitivity_n':int(s.loc[task,'n']),'primary_accuracy':float(p.loc[task,'accuracy']),
                               'sensitivity_accuracy':float(s.loc[task,'accuracy']),
                               'accuracy_difference_sensitivity_minus_primary':float(s.loc[task,'accuracy']-p.loc[task,'accuracy']),
                               'raw_weight_r':r,'intersection_voxels':n,'resampled_for_comparison':resampled})
    keys=[k for k in patterns if k[2]=='common']
    for i,ka in enumerate(keys):
        for kb in keys[i+1:]:
            for kind in ('weight','haufe','mean_difference'):
                r,n,resampled=intersection_correlation(patterns[ka][kind],*geometries[ka[0]],patterns[kb][kind],*geometries[kb[0]])
                cross.append({'family_a':ka[1],'family_b':kb[1],'kind':kind,'spatial_r':r,'intersection_voxels':n,
                              'resampled_for_comparison':resampled})
    write_tsv(out,'results/aggregate/qc_robustness.tsv',robust)
    write_tsv(out,'results/aggregate/cross_family_similarity.tsv',cross)
    return pd.DataFrame(robust),pd.DataFrame(cross)


def read_products(out,scopes):
    """Reload small completed characterization maps for plotting without redoing fits."""
    patterns={}; stability={}; geometries={}
    for cohort,families in TARGETS.items():
        scope=scopes[(PRIMARY,cohort)]; mask,ref=geometry(scope); geometries[cohort]=(mask,ref)
        def read(stem):
            assert_development(scope)
            p=out.output('results/maps/'+stem+'.nii.gz')
            image=nib.load(p)
            require(image.shape==mask.shape and np.allclose(image.affine,ref.affine),'characterization map geometry mismatch')
            return image.get_fdata(dtype=np.float32)[mask]
        for family in families:
            for name in (*TASKS[cohort],'common'):
                stem=cohort+'_'+family+'_'+name
                patterns[(cohort,family,name)]={kind:read(stem+suffix) for kind,suffix in
                      [('weight','_raw_weights'),('haufe','_haufe_pattern'),('mean_difference','_mean_difference')]}
            stability[(cohort,family)]={kind:{stat:read(cohort+'_'+family+'_bootstrap_'+kind+'_'+stat)
                                            for stat in ('mean','sd','proportion_positive','proportion_negative','sign_stable')}
                                        for kind in ('weight','haufe')}
    return patterns,stability,geometries


def render(out,scopes,spec):
    from characterization_figures import (brain_figures,fixed_figures,similarity_figures,main_pattern_figures,
                                         expression_figures,reliability_figures,qc_figure,permutation_figures)
    patterns,stability,geometries=read_products(out,scopes)
    fixed_figures(out,scopes)
    similarity_figures(out,patterns)
    for (cohort,family,name),maps in patterns.items():
        mask,ref=geometries[cohort]; stem=cohort+'_'+family+'_'+name
        for kind,label in [('weight','Raw SVM discriminative coefficients'),('haufe','Haufe descriptive activation pattern'),
                           ('mean_difference','Descriptive mean condition difference')]:
            brain_figures(out,stem+'_'+kind,reconstruct(maps[kind],mask),mask,ref,family,label,spec)
        if name=='common':
            for kind in ('weight','haufe'):
                brain_figures(out,stem+'_bootstrap_'+kind+'_sign_stable',
                              reconstruct(stability[(cohort,family)][kind]['sign_stable'],mask),mask,ref,family,
                              'Bootstrap '+kind+' sign-stable for descriptive visualization',spec)
    main_pattern_figures(out,scopes,patterns,stability,geometries)
    expression_figures(out,scopes); reliability_figures(out,scopes)
    qc_figure(out,pd.read_csv(out.output('results/aggregate/qc_robustness.tsv'),sep='\t'))
    nulls={}
    for cohort,families in TARGETS.items():
        for family in families:
            for mode in (('common','lopo') if family=='social_context' else ('common',)):
                state=json.loads(out.output('work/permutation/'+cohort+'_'+family+'_'+mode+'.json').read_text())
                nulls[(cohort,family,mode)]=(TASKS[cohort],np.array(state['null']))
    permutation_figures(out,pd.read_csv(out.output('results/aggregate/permutation_summary.tsv'),sep='\t'),nulls)
    from characterization_report import report
    report(out,scopes,spec)


def run(base,dry_run=False,plots_only=False,spec=None):
    spec=settings(base) if spec is None else spec
    out=output_config(base)
    with run_lock(base):
        print('Characterization audit: frozen v4 code/results and private participant membership',flush=True)
        try: scopes=audit(base,spec)
        except Exception as exc:
            write_audit(out,'**NOT PASSED. No new scientific outputs are authorized.**\n\n'+str(exc)+'\n\nStatic review found no evident supervised leakage; the complete runtime/private-file audit has not passed. This is not a clean audit certification.')
            raise
        before=frozen_snapshot(base)
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','nibabel','pandas','matplotlib']}
        key=digest({'snapshot':before,'settings':spec,'compute_code':sha256(base.root/'code/characterization_compute.py'),
                    'orchestration_code':sha256(base.root/'code/characterize_v4.py'),
                    'audit_code':sha256(base.root/'code/characterization_audit.py'),
                    'software':versions})
        path=out.output('work/frozen_inputs.json')
        if path.exists(): require(json.loads(path.read_text())['fingerprint']==key,'original inputs/software/computation changed since checkpoint')
        write_audit(out,'**Membership and static implementation checks PASSED.** No leakage problem found in recorded membership or preprocessing. Numerical OOF reconstruction is pending; no scientific outputs have yet been authorized by this audit stage.')
        if dry_run:
            print('CHARACTERIZATION DRY RUN PASSED: membership/hash audit only; no voxel reads, models, or scientific outputs. OOF numerical check remains pending.',flush=True)
            return
        write_json(out,'work/frozen_inputs.json',{'fingerprint':key,'snapshot':before})
        write_json(out,'provenance/characterization.json',{
                   'fingerprint':key,'source_snapshot_sha256':digest(before),'settings':spec,
                   'software_versions':versions,'holdout_scored':False,
                   'original_model_software':{cohort:scopes[(PRIMARY,cohort)]['model']['software_versions'] for cohort in TARGETS},
                   'source_sample_manifest_sha256':scopes[(PRIMARY,'partner_pair')]['model']['sample_manifest_sha256'],
                   'bootstrap_unit':'Whole participants, including all model tasks/classes; 1000 deterministic resamples',
                   'permutation_scheme':'One independent sign per participant, shared across tasks and applied to training and test labels; full five-fold refitting',
                   'haufe_definition':'Cov(X, Xw+b) / Var(Xw+b), exact spatially centered development training maps; no feature z-scoring',
                   'cluster_definition':'Separate positive/negative 6-neighbor components of bootstrap-mean Haufe values with sign proportion >= .975; minimum 20 voxels; descriptive only'})
        complete=out.output('work/computation_complete.json')
        if plots_only: require(complete.exists(),'--plots-only requires completed audited computations')
        stage='OOF verification'
        write_json(out,'provenance/run_status.json',{'status':'in_progress','stage':stage,'fingerprint':key,'holdout_scored':False})
        try:
            if not complete.exists():
                # Build both primary caches and verify every selected OOF model before scientific outputs.
                all_arrays={}; geometries={}; checks=[]
                for cohort in TARGETS:
                    scope=scopes[(PRIMARY,cohort)]
                    arrays,mask,ref=build_arrays(out,scope,cohort,key)
                    all_arrays[cohort]=arrays; geometries[cohort]=(mask,ref)
                    checks.extend(verify_oof(out,scope,cohort,arrays,mask,ref,spec))
                write_audit(out,'**PASSED.** No leakage/preprocessing error found in the audited code and recorded private membership. Saved float32 fold maps reproduce selected binary OOF margins and classifications within the prespecified numerical tolerance. All validation participants remain excluded.')
                write_tsv(out,'results/aggregate/oof_reconstruction_checks.tsv',checks)
                patterns={}; correlations=[]; cluster_rows=[]; permutation_rows=[]
                stage='maps and resampling'
                for cohort,families in TARGETS.items():
                    scope=scopes[(PRIMARY,cohort)]; arrays=all_arrays[cohort]; mask,ref=geometries[cohort]
                    p,rows=descriptive_maps(out,scope,cohort,arrays,mask,ref); patterns.update(p); correlations.extend(rows)
                    for family in families:
                        stats=bootstrap(out,scope,cohort,family,arrays[family],mask,ref,spec,key)
                        cluster_rows.extend({'cohort':cohort,'family':family,**r} for r in
                                            clusters(stats['haufe']['sign_stable'],mask,ref,spec['minimum_cluster_voxels']))
                        for mode in (('common','lopo') if family=='social_context' else ('common',)):
                            rows,_=permutation(out,scope,cohort,family,arrays[family],spec,key,mode)
                            permutation_rows.extend(rows)
                write_tsv(out,'results/aggregate/weight_pattern_correlations.tsv',correlations)
                write_tsv(out,'results/aggregate/descriptive_clusters.tsv',cluster_rows,
                          columns=['cohort','family','sign','voxel_count','peak_absolute_haufe','peak_signed_haufe','x','y','z','mean_haufe','connectivity','inferential_p'])
                write_tsv(out,'results/aggregate/permutation_summary.tsv',permutation_rows)
                spatial_comparisons(out,scopes,patterns,geometries)
                require(frozen_snapshot(base)==before,'original v4 outputs changed during computation')
                products={str(p.relative_to(out.root)):sha256(p) for p in out.output('results').rglob('*') if p.is_file()}
                # Private OOF scores and permutation checkpoints are also immutable plot inputs.
                for p in out.output('work').rglob('*'):
                    if p.is_file() and (p.name.endswith('_oof_scores.tsv') or p.parent.name=='permutation'):
                        products[str(p.relative_to(out.root))]=sha256(p)
                write_json(out,'work/computation_complete.json',{'fingerprint':key,'products':products})
            saved=json.loads(complete.read_text()); require(saved['fingerprint']==key,'completed computation fingerprint mismatch')
            for rel,expected in saved['products'].items(): require(sha256(base.root/rel)==expected,'computed plot input changed')
            write_audit(out,'**PASSED.** Private membership/source-load records and frozen implementation passed. Saved float32 fold maps reproduced selected binary OOF margins and classifications before characterization. No leakage/preprocessing error found. Validation remains untouched.')
            stage='plotting and report'
            render(out,scopes,spec)
            require(frozen_snapshot(base)==before,'original v4 outputs changed during plotting')
            perm=pd.read_csv(out.output('results/aggregate/permutation_summary.tsv'),sep='\t')
            centered=not perm.chance_centering_flag.any()
            write_json(out,'provenance/run_status.json',{'status':'complete' if centered else 'needs_review',
                       'fingerprint':key,'holdout_scored':False,'validation_n':50,'original_v4_hashes_unchanged':True,
                       'permutation_nulls_chance_centered':centered,'bootstrap_samples':spec['bootstrap_samples'],
                       'permutations':spec['permutations'],'note':'Null centering uses descriptive |mean - .5| <= .05, not an inferential test.'})
            for name in ['Figure1_schematic','Figure2_context_reward_transfer','Figure3_social_context_pattern',
                         'Figure4_closeness_valence','Figure5_oof_expression','Figure6_run_reliability']:
                print(out.output('results/figures/'+name+'.png'),flush=True)
            print(out.output('reports/REPORT.md'),flush=True)
            for name in ['weight_pattern_correlations','task_pattern_similarity','cross_family_similarity','qc_robustness','permutation_summary']:
                print(out.output('results/aggregate/'+name+'.tsv'),flush=True)
            print(perm[['family','scope','test_task','observed_accuracy','null_mean','empirical_p']].to_string(index=False),flush=True)
            print('Validation N=50 remains untouched; holdout_scored = False.',flush=True)
            if not centered: print('REVIEW REQUIRED: one or more permutation null means depart from .50 by more than .05. Do not claim sanity checks passed.',flush=True)
        except Exception:
            unchanged=frozen_snapshot(base)==before
            write_json(out,'provenance/run_status.json',{'status':'failed','stage':stage,'holdout_scored':False,
                       'original_v4_hashes_unchanged':unchanged,'note':'Checkpointed computations retained. No automatic changes to v4.'})
            if stage=='OOF verification': write_audit(out,'**STOPPED during numerical OOF verification.** Do not interpret or visualize characterization results. Inspect the local error; no automatic repair was attempted.')
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true',help='Audit hashes/membership only; no voxel reads')
    group.add_argument('--plots-only',action='store_true',help='Re-render completed checked computations without resampling')
    args=parser.parse_args(argv)
    try: run(load_config(),args.dry_run,args.plots_only)
    except (PipelineError,FileNotFoundError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr); return 1
    return 0

if __name__=='__main__': raise SystemExit(main())
