"""Final prespecified development follow-up; protected validation is never scored."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from pilot import run_lock
from utils import load_config, PipelineError, sha256, write_json, write_text, write_tsv
from characterization_audit import audit, require, digest, PRIMARY, SENSITIVITY, PARAMETERS, SEED, TASKS
from cross_valence_design import output_config, plans, permutation_plans, training_names
from cross_valence_compute import build_arrays, cv_margins, summaries, final_patterns, permute
from cross_valence_anatomy import inventory, prepare


def snapshot(base):
    """Existing public aggregates plus an allowlist of private metadata, never private voxels."""
    result={}
    for tree in ('results','reports','provenance'):
        for p in sorted((base.root/tree).rglob('*')):
            rel=p.relative_to(base.root)
            if not p.is_file() or tuple(rel.parts[1:3])==('revised','cross_valence'): continue
            result[str(rel)]=sha256(p)
    private=['work/splits/subject_split_v1.tsv','work/splits/development_folds.tsv','work/revised/samples/manifest.tsv']
    for policy in (PRIMARY,SENSITIVITY):
        for cohort in TASKS:
            private.extend(f'work/revised/{policy}/{cohort}/{name}' for name in
                ('model_membership.tsv','oof_predictions.tsv','run_predictions.tsv','source_image_metrics.tsv','map_norms.tsv','diagnostics/mask_sources.tsv'))
    for rel in private:
        if (base.root/rel).is_file(): result[rel]=sha256(base.root/rel)
    return result


def settings(base):
    spec=yaml.safe_load((base.root/'config/cross_valence.yaml').read_text())
    require(spec['seed']==SEED and spec['permutations']==500 and spec['bootstrap_samples']==10000,'prespecified resampling settings changed')
    require(spec['classifier']==PARAMETERS,'frozen classifier changed')
    spec['frozen_outputs']=json.loads((base.root/'config/cross_valence_frozen_outputs.json').read_text())
    return spec


def audit_text(out,phase):
    write_text(out,'reports/IMPLEMENTATION_AUDIT.md','# Cross-valence implementation audit\n\n'+phase+'\n\n'
        'Reuse of the v4 private membership/source-load audit checks participant separation across every task and valence, original protected assignments, fixed folds and masks, exact saved model hashes, QC, parameters and source logs. '
        'The completed characterization status and frozen public output hashes are checked before new computations. '
        'Original label-free development coverage masks include later CV test participants and are not fold-specific; they are retained. '
        'No feature selection, population normalization, hyperparameter tuning, visual exclusion fitting or validation scoring is implemented. '
        'Each new source-image read and model operation requires the development guard. All models use C=1, dual=True, class_weight=None, max_iter=100000, tol=.0001, random_state=20260928. '
        'Numerical reconstruction of existing context and same-valence friend–stranger OOF predictions must pass before new scientific outputs. '
        'Private fold checkpoints record exact train/test people and training domains. Combined-task endpoints average correctness within participant; bootstrap/permutation units are participants. '
        'A single permuted label flip is shared across all of each participant’s maps, including test labels. '
        'This audits recorded pipeline behavior, not unrecorded manual access.\n\nholdout_scored=False.\n')


def run(base,dry_run=False,plots_only=False,spec=None,workers=96):
    spec=settings(base) if spec is None else spec; out=output_config(base)
    stage='audit'; before=None
    with run_lock(base):
        try:
            print('Cross-valence: auditing frozen v4 and completed characterization',flush=True)
            scopes=audit(base,spec)
            char=json.loads((base.root/'provenance/revised/characterization/run_status.json').read_text())
            require(char['status']=='complete' and char['holdout_scored'] is False and char['original_v4_hashes_unchanged'],
                    'completed, unscored characterization required')
            before=snapshot(base)
            for rel,value in spec['frozen_outputs'].items(): require(before.get(rel)==value,'frozen output changed: '+rel)
            require(char['fingerprint']==spec['characterization_fingerprint'],'characterization fingerprint changed')
            audit_text(out,'Membership, configuration and frozen output checks PASSED. Source reconstruction/new fold execution pending.')
            if dry_run:
                print('DRY RUN PASSED: metadata/hash checks only; no NIfTI load, fitting, or validation scoring.',flush=True)
                return
            assets=inventory(base)
            require(assets['background'] is not None,'No local standard anatomical T1 background found. Set FSLDIR to an existing installation; no download is attempted.')
            versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','pandas','nibabel','matplotlib')}
            code={p.name:sha256(p) for p in sorted((base.root/'code').glob('cross_valence*.py'))}
            key=digest(dict(snapshot=before,spec=spec,code=code,software=versions))
            marker=out.output('work/frozen_inputs.json')
            if marker.exists(): require(json.loads(marker.read_text())['fingerprint']==key,'inputs/code/software changed since checkpoint')
            write_json(out,'work/frozen_inputs.json',dict(fingerprint=key,snapshot=before))
            def status(stage,status='in_progress',**extra):
                write_json(out,'provenance/run_status.json',dict(status=status,stage=stage,fingerprint=key,holdout_scored=False,validation_n=50,**extra))
                print('Cross-valence stage: '+stage,flush=True)
            status('source and OOF verification'); stage='source and OOF verification'
            complete=out.output('work/computation_complete.json')
            require(not plots_only or complete.exists(),'--plots-only requires completed computations')
            if not complete.exists():
                arrays={}; geometries={}; checks=[]
                for cohort in ('partner_pair','three_paradigm'):
                    x,mask,ref,rows=build_arrays(out,scopes[(PRIMARY,cohort)],cohort,key,spec)
                    arrays[cohort]=x; geometries[cohort]=(mask,ref); checks.extend(rows)
                audit_text(out,'PASSED: private membership, immutable inputs and numerical original OOF reconstruction. New models retain the checked fold and guard rules.')
                write_tsv(out,'results/aggregate/oof_reconstruction_checks.tsv',checks)
                stage='development cross-valence fits'; status(stage)
                rows=[]; predictions=[]; members=[]
                for cohort in ('partner_pair','three_paradigm'):
                    scope=scopes[(PRIMARY,cohort)]
                    for p in plans(cohort):
                        margins,membership=cv_margins(out,scope,cohort,arrays[cohort],p,key)
                        rows.extend(summaries(cohort,p,margins,spec)); members.extend(membership)
                        predictions.extend(dict(cohort=cohort,family=p['family'],scope=p['scope'],train_domain=p['name'],test_domain=name,
                            subject=s,fold=scope['folds'][s],margin=float(margins[i,j]),correct=bool(margins[i,j]>0))
                            for i,s in enumerate(scope['subjects']) for j,name in enumerate(p['test_names']))
                        print(f'  Finished {cohort} {p["family"]} {p["name"]}',flush=True)
                # Reuse original collapsed-context OOF predictions; never use in-sample DEV accuracy.
                scope=scopes[(PRIMARY,'partner_pair')]
                baseline=scope['predictions'].query('family == "social_context" and scope == "common"')
                baseline_margins=np.column_stack([baseline[baseline.test_task==task].set_index('subject').loc[scope['subjects'],'margin'] for task in TASKS['partner_pair']])
                p=dict(family='social_context',name='collapsed_common',scope='collapsed',test_names=list(TASKS['partner_pair']))
                rows.extend(summaries('partner_pair',p,baseline_margins,spec))
                predictions.extend(dict(cohort='partner_pair',family='social_context',scope='collapsed',train_domain='collapsed_common',test_domain=task,
                    subject=s,fold=scope['folds'][s],margin=float(baseline_margins[i,j]),correct=bool(baseline_margins[i,j]>0))
                    for i,s in enumerate(scope['subjects']) for j,task in enumerate(TASKS['partner_pair']))
                perf=pd.DataFrame(rows)
                write_tsv(out,'work/oof_predictions.tsv',predictions); write_tsv(out,'work/model_membership.tsv',members)
                write_tsv(out,'results/aggregate/performance.tsv',perf)
                for cohort,family,size,name in [('partner_pair','social_context',4,'primary_social_context_matrix'),
                    ('partner_pair','friend_stranger_context',4,'primary_friend_stranger_matrix'),
                    ('three_paradigm','social_context',6,'doors_boundary_matrix')]:
                    selection=perf[(perf.cohort==cohort)&(perf.family==family)&(perf.scope=='matrix')]
                    require(len(selection)==size*size,'incomplete decoding matrix')
                    write_tsv(out,'results/aggregate/'+name+'.tsv',selection)
                stage='final development patterns'; status(stage)
                models=final_patterns(out,scope,arrays['partner_pair'],*geometries['partner_pair'],spec,key)
                comparisons=scope['performance'].query('family in ["social_context", "valence", "social_reward", "closeness_reward"]')
                write_tsv(out,'results/aggregate/frozen_v4_comparators.tsv',comparisons)
                stage='participant permutations'; status(stage); permutation_rows=[]
                from cross_valence_parallel import run_permutations
                all_nulls=run_permutations(out,scope,arrays['partner_pair'],spec,key,workers)
                for p in permutation_plans():
                    null=all_nulls[p['family']+'_'+p['name']]
                    targets=['combined'] if p['scope']=='pooled_cross_valence' else p['test_names']
                    for j,target in enumerate(targets):
                        observed=perf[(perf.cohort=='partner_pair')&(perf.family==p['family'])&(perf.train_domain==p['name'])&(perf.test_domain==target)]
                        require(len(observed)==1,'missing or duplicated permutation endpoint')
                        accuracy=float(observed.accuracy.iloc[0]); values=null[:,j]
                        permutation_rows.append(dict(family=p['family'],train_domain=p['name'],test_domain=target,n=len(scope['subjects']),
                            permutations=spec['permutations'],observed_accuracy=accuracy,null_mean=float(values.mean()),null_sd=float(values.std(ddof=1)),
                            null_q025=float(np.quantile(values,.025)),null_q975=float(np.quantile(values,.975)),
                            empirical_p=float((1+(values>=accuracy).sum())/(len(values)+1)),chance_centering_flag=abs(values.mean()-.5)>.05))
                    # Public summaries accumulate atomically; private completed permutations are resumable.
                    write_tsv(out,'results/aggregate/permutation_summary.tsv',permutation_rows)
                require(snapshot(base)==before,'original v4/characterization artifacts changed')
                products={str(p.relative_to(base.root)):sha256(p) for p in out.output('results').rglob('*') if p.is_file()}
                products[str(out.output('provenance/models.json').relative_to(base.root))]=sha256(out.output('provenance/models.json'))
                for p in out.output('work/permutations').glob('*.json'): products[str(p.relative_to(base.root))]=sha256(p)
                write_json(out,'work/computation_complete.json',dict(fingerprint=key,products=products))
            saved=json.loads(complete.read_text()); require(saved['fingerprint']==key,'computation fingerprint mismatch')
            for rel,value in saved['products'].items(): require(sha256(base.root/rel)==value,'completed computation product changed: '+rel)
            stage='anatomical presentation and reports'; status(stage)
            proposal=prepare(out,base,scopes,assets)
            from cross_valence_report import render
            render(out,scopes,spec,assets)
            require(snapshot(base)==before,'original artifacts changed during reporting')
            perm=pd.read_csv(out.output('results/aggregate/permutation_summary.tsv'),sep='\t')
            centered=not perm.chance_centering_flag.any()
            write_json(out,'provenance/analysis.json',dict(fingerprint=key,software=versions,code_sha256=code,
                seed=SEED,classifier=PARAMETERS,permutations=spec['permutations'],bootstrap_samples=spec['bootstrap_samples'],
                source_snapshot_sha256=digest(before),holdout_scored=False,visual_exclusion_executed=False,
                visual_proposal_status=proposal['status'],alternate_qc_new_models_run=False))
            audit_text(out,'PASSED: audited membership, original OOF reconstruction and all guarded development fits completed; original v4 and characterization output hashes unchanged.')
            status('finished','complete' if centered else 'needs_review',original_outputs_unchanged=True,permutation_nulls_chance_centered=bool(centered),visual_exclusion_executed=False)
            print(perm[['family','train_domain','test_domain','observed_accuracy','null_mean','empirical_p']].to_string(index=False),flush=True)
            perf=pd.read_csv(out.output('results/aggregate/performance.tsv'),sep='\t')
            print(perf[(perf.cohort=='partner_pair')&perf.scope.isin(['pooled_cross_valence','collapsed'])][['family','train_domain','test_domain','accuracy']].to_string(index=False),flush=True)
            print(pd.read_csv(out.output('results/aggregate/spatial_similarity.tsv'),sep='\t').query('kind == "haufe"').to_string(index=False),flush=True)
            for entry in json.loads(out.output('provenance/models.json').read_text())['models']:
                if entry['mode']=='collapsed' and entry['model']=='common': print(entry['weight_path'],entry['weight_sha256'],flush=True)
            for name in ('REPORT','FREEZE_CANDIDATES','VISUAL_SENSITIVITY_PROPOSAL'): print(out.output('reports/'+name+'.md'),flush=True)
            print('Validation N=50 untouched; holdout_scored = False. Visual exclusion NOT executed.',flush=True)
        except Exception as exc:
            if stage in ('audit','source and OOF verification'): audit_text(out,'NOT PASSED. Stop scientific interpretation. '+str(exc))
            write_json(out,'provenance/run_status.json',dict(status='failed',stage=stage,holdout_scored=False,
                original_outputs_unchanged=(snapshot(base)==before) if before is not None else None))
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true'); group.add_argument('--plots-only',action='store_true')
    parser.add_argument('--workers',type=int,default=96,help='Requested permutation processes (default 96); bounded by CPU availability and estimated RAM')
    args=parser.parse_args(argv)
    try: run(load_config(),args.dry_run,args.plots_only,workers=args.workers)
    except (PipelineError,FileNotFoundError) as exc:
        print('ERROR: '+str(exc)); return 1
    return 0

if __name__=='__main__': raise SystemExit(main())
