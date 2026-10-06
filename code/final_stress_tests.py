"""Last development stress tests; no validation-loading/scoring option exists."""
import argparse,json,importlib.metadata,re,uuid
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from utils import load_config,PipelineError,sha256,write_json,write_text,write_tsv
from pilot import run_lock
from characterization_audit import audit,require,digest,PRIMARY
from characterization_compute import assert_development
from cross_valence import settings as prior_settings
from final_stress_design import output_config,make_plans,usable
from final_stress_sources import inventory,build_features,visual_mask
from final_stress_compute import run_plan,summarize,fixed_probes,specificity,spatial_comparison,valence_controls,verify_baseline


def snapshot(base):
    result={}
    for tree in ('results','reports','provenance'):
        for p in sorted((base.root/tree).rglob('*')):
            rel=p.relative_to(base.root)
            if p.is_file() and tuple(rel.parts[1:3])!=('revised','final_stress_tests'): result[str(rel)]=sha256(p)
    for rel in ('work/splits/subject_split_v1.tsv','work/splits/development_folds.tsv','work/revised/samples/manifest.tsv'):
        p=base.root/rel
        if p.exists(): result[rel]=sha256(p)
    return result

def verify_contract(base):
    frozen=json.loads((base.root/'config/final_stress_frozen.json').read_text())
    for rel,h in frozen.items(): require((base.root/rel).is_file() and sha256(base.root/rel)==h,'frozen artifact/code changed: '+rel)
    source=json.loads((base.root/'config/final_stress_sources.json').read_text())
    for repository,files in source['runtime_sources'].items():
        for rel,h in files.items(): require(sha256(base.repos[repository]/rel)==h,'audited source changed: '+repository+'/'+rel)
    cross=json.loads((base.root/'provenance/revised/cross_valence/run_status.json').read_text())
    require(cross['status']=='complete' and cross['holdout_scored'] is False and cross['original_outputs_unchanged'],'completed unscored cross-valence run required')
    return frozen,source

def prepare_identity(out,key,before,eligible,preview=False):
    """Archive only known design-preview artifacts on an explicit preview rerun."""
    marker=out.output('work/identity.json')
    if marker.exists() and json.loads(marker.read_text())['fingerprint']!=key:
        message='stress-test checkpoint inputs changed; only a design-only --fairness-preview restart can archive automatically; fitted checkpoints require review'
        require(preview,message)
        work=out.output('work')
        require(all(p.name in {'identity.json','inventory.tsv','fairness','archived_previews'} for p in work.iterdir()),message)
        fairness=out.output('work/fairness')
        # An allowlist rejects FEAT outputs, completion markers, unknown products and symlinks.
        ev=r'(?:(?:non)?social_(?:unfair|fair|preoffer)|endowment_(?:high|low)|rt_constant|rt_pmod|missed_trial|missed_feedback)\.txt'
        artifact=r'(?:identity\.json|render\.log|trial_counts_detail\.tsv|design(?:_cov)?\.(?:fsf|mat|con|min|trg|frf|png|ppm)|'+ev+r')'
        for p in [fairness,*fairness.rglob('*')] if fairness.exists() else []:
            require(not p.is_symlink(),message)
            rel=p.relative_to(fairness).as_posix()
            allowed=(rel=='.' or bool(re.fullmatch(r'sub-[A-Za-z0-9]+(?:/ses-01(?:/run-[12])?)?',rel))) if p.is_dir() else (
                rel=='design_qc.tsv' or bool(re.fullmatch(r'sub-[A-Za-z0-9]+/ses-01/run-[12]/'+artifact,rel)))
            require(allowed,message)
        public={
            'results':{'aggregate/inventory.tsv','aggregate/phase_directory_inventory.tsv','aggregate/phase_retained_runs.tsv','figures/fairness_design_preview.png','figures/fairness_design_preview.pdf'},
            'provenance':{'run_status.json','fairness_status.json'},
        }
        for tree,allowed in public.items():
            root=out.output(tree)
            for p in root.rglob('*'):
                require(not p.is_symlink() and (p.is_dir() or p.relative_to(root).as_posix() in allowed),message)
        archive='work/archived_previews/'+uuid.uuid4().hex
        # Preserve prior previews privately. Move the root identity last so interruptions remain guarded.
        for rel in ('work/fairness','results/figures/fairness_design_preview.png','results/figures/fairness_design_preview.pdf',
                    'provenance/fairness_status.json','provenance/run_status.json','reports/FAIRNESS_DESIGN_QC.md','work/identity.json'):
            source=out.output(rel)
            if source.exists():
                destination=out.output(archive+'/'+rel); destination.parent.mkdir(parents=True,exist_ok=True)
                source.rename(destination)
        print('Archived prior design-only preview in '+str(out.output(archive))+'; rendering again with current inputs.',flush=True)
    write_json(out,'work/identity.json',dict(fingerprint=key,original=before,eligible=eligible))

def run(base,workers=96,dry_run=False,fairness_preview=False,plots_only=False):
    require(isinstance(workers,int) and workers>0,'workers must be a positive integer')
    out=output_config(base); before=None; stage='audit'
    with run_lock(base):
        try:
            frozen,sources=verify_contract(base); scopes=audit(base,prior_settings(base)); scope=scopes[(PRIMARY,'partner_pair')]
            before=snapshot(base); assert_development(scope)
            # Add UGR only to this downstream read adapter; never mutate v4 config/files.
            scope={**scope,'c':deepcopy(scope['c'])}
            scope['c'].repos['ugr']=base.repos['ugr']
            scope['c'].contrasts['ugr']=deepcopy(base.contrasts['ugr'])
            if dry_run:
                print('DRY RUN PASSED: frozen artifacts, source code, private sample/holdout/fold audit. No new images loaded. holdout_scored = False',flush=True); return
            eligible,phase_paths=inventory(out,scope)
            require(usable(scope,eligible['baseline']),'current-QC baseline has fewer than ten people or is missing original folds; see inventory')
            if not fairness_preview:
                require(usable(scope,eligible['sr_outcome']),'primary SR outcome analysis requires >=10 eligible development participants and all five frozen folds; stop before feature loading, see inventory')
            code={p.name:sha256(p) for p in (base.root/'code').glob('final_stress*.py')}
            software={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','nibabel','pandas')}
            qc=base.repos['linux2']/base.paths['qc_table']
            key=digest(dict(original=before,code=code,software=software,sources=sources,qc=sha256(qc),eligible=eligible))
            prepare_identity(out,key,before,eligible,preview=fairness_preview)
            for name in ('aggregate','figures','maps'): out.output('results/'+name).mkdir(parents=True,exist_ok=True)
            def status(stage,state='in_progress',**extra):
                write_json(out,'provenance/run_status.json',dict(status=state,stage=stage,fingerprint=key,holdout_scored=False,validation_n=50,**extra))
                print('Final stress tests: '+stage,flush=True)
            from final_stress_fairness import prepare_and_fit,categorical_features
            if fairness_preview:
                require(eligible['ugr'],'no QC-qualified development UGR sample for preview')
                stage='fairness design preview'; status(stage)
                result=prepare_and_fit(out,scope,eligible['ugr'],key,workers,preview=True)
                require(snapshot(base)==before,'frozen outputs changed')
                status(stage,result['status'],original_outputs_unchanged=True)
                print(out.output('reports/FAIRNESS_DESIGN_QC.md'),flush=True); print('Preview only. holdout_scored = False',flush=True); return
            stage='source features'; status(stage)
            arrays,available,mask,ref=build_features(out,scope,eligible,phase_paths,key)
            verify_baseline(out,scope,arrays,available,mask,ref)
            keep=visual_mask(out,scope,mask,ref)
            complete=out.output('work/computation_complete.json')
            require(not plots_only or complete.exists(),'--plots-only requires completed computations')
            if not complete.exists():
                plans=[p for p in make_plans(available) if p['group']!='norm']; skipped=[]; accepted=[]; records=[]; members=[]
                stage='phase, architecture and outcome-to-UGR CV'; status(stage)
                for p in plans:
                    if not usable(scope,p['subjects']):
                        skipped.append(dict(model=p['name'],visual=p['visual'],n=len(p['subjects']),reason='needs >=10 matched development participants and all five frozen folds')); continue
                    rows,mm=run_plan(out,scope,arrays,p,keep,key,mask,ref); records.extend(rows); members.extend(mm); accepted.append(p)
                    print('  Finished '+p['name']+(' visual excluded' if p['visual'] else ''),flush=True)
                records.extend(fixed_probes(scope,arrays,available,mask,ref)); records.extend(specificity(out,scope,arrays,accepted,keep))
                write_tsv(out,'work/predictions.tsv',records); write_tsv(out,'work/model_membership.tsv',members)
                write_tsv(out,'results/aggregate/performance.tsv',summarize(records))
                stage='categorical UGR design gate and exploratory fits'; status(stage)
                if usable(scope,eligible['ugr']): branch=prepare_and_fit(out,scope,eligible['ugr'],key,workers)
                else:
                    branch=dict(status='stopped_insufficient_development_sample',reason='No computationally adequate UGR sample in the frozen development pool',holdout_scored=False)
                    write_json(out,'provenance/fairness_status.json',branch)
                if branch['status']=='complete':
                    arrays.update(categorical_features(out,scope,eligible['ugr'],mask,ref))
                    available.update({k:eligible['ugr'] for k in ('ugr_norm','ugr_computer_norm')})
                    for p in make_plans(available):
                        if p['group']!='norm': continue
                        if not usable(scope,p['subjects']):
                            skipped.append(dict(model=p['name'],visual=False,n=len(p['subjects']),reason='insufficient matched norm sample')); continue
                        rows,mm=run_plan(out,scope,arrays,p,keep,key,mask,ref); records.extend(rows); members.extend(mm); accepted.append(p)
                    records.extend(valence_controls(scope,arrays,available,mask,ref))
                perf=summarize(records)
                write_tsv(out,'work/predictions.tsv',records); write_tsv(out,'work/model_membership.tsv',members)
                write_tsv(out,'results/aggregate/performance.tsv',perf)
                spatial_comparison(out,scope,mask,ref,accepted,keep)
                stage='participant permutations'; status(stage)
                from final_stress_parallel import run as permutations
                results=permutations(out,scope,arrays,accepted,keep,key,workers); rows=[]
                for model,target,values in results:
                    g=perf[perf.model.eq(model)&perf.test.eq(target)]; require(len(g)==1,'permutation target not unique')
                    observed=float(g.accuracy.iloc[0]); rows.append(dict(model=model,test=target,n=int(g.n.iloc[0]),permutations=len(values),observed_accuracy=observed,
                        null_mean=float(values.mean()),null_sd=float(values.std(ddof=1)),empirical_p=float((1+(values>=observed).sum())/(len(values)+1)),chance_centering_flag=bool(abs(values.mean()-.5)>.05)))
                write_tsv(out,'results/aggregate/permutation_summary.tsv',rows,columns=['model','test','n','permutations','observed_accuracy','null_mean','null_sd','empirical_p','chance_centering_flag'])
                require(snapshot(base)==before,'original results or frozen candidates changed')
                products={str(p.relative_to(base.root)):sha256(p) for p in out.output('results').rglob('*') if p.is_file()}
                for name in ('predictions.tsv','partner_expression.tsv','model_membership.tsv'):
                    p=out.output('work/'+name)
                    if p.exists(): products[str(p.relative_to(base.root))]=sha256(p)
                write_json(out,'work/computation_complete.json',dict(fingerprint=key,products=products,skipped=skipped,fairness=branch))
            saved=json.loads(complete.read_text()); require(saved['fingerprint']==key,'completed fingerprint changed')
            for rel,h in saved['products'].items(): require(sha256(base.root/rel)==h,'completed product changed: '+rel)
            perf=pd.read_csv(out.output('results/aggregate/performance.tsv'),sep='\t'); perms=pd.read_csv(out.output('results/aggregate/permutation_summary.tsv'),sep='\t')
            stage='figures and interpretation'; status(stage)
            from final_stress_report import render
            render(out,perf,perms,saved['fairness'],keep,saved['skipped'])
            require(snapshot(base)==before,'frozen artifacts changed during reporting')
            centered=not perms.chance_centering_flag.any(); all_required=not saved['skipped']
            write_json(out,'provenance/analysis.json',dict(code_sha256=code,software=software,fingerprint=key,original_outputs_unchanged=True,
                holdout_scored=False,visual_exclusion_role='descriptive_only_no_selection_no_success_requirement',fairness_status=saved['fairness']['status']))
            status('finished','complete' if centered and all_required else 'needs_review',original_outputs_unchanged=True,permutation_nulls_chance_centered=bool(centered),fairness_status=saved['fairness']['status'])
            print(perf[['model','test','n','accuracy','evaluation']].to_string(index=False),flush=True)
            for name in ('REPORT','COLLABORATOR_EMAIL_DRAFT','TASK_PHASE_AUDIT'): print(out.output('reports/'+name+'.md'),flush=True)
            print('Frozen candidates unchanged. holdout_scored = False',flush=True)
        except Exception:
            write_json(out,'provenance/run_status.json',dict(status='failed',stage=stage,holdout_scored=False,original_outputs_unchanged=snapshot(base)==before if before is not None else None))
            raise

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group()
    g.add_argument('--dry-run',action='store_true'); g.add_argument('--fairness-preview',action='store_true'); g.add_argument('--plots-only',action='store_true')
    p.add_argument('--workers',type=int,default=96)
    a=p.parse_args(argv)
    try: run(load_config(),a.workers,a.dry_run,a.fairness_preview,a.plots_only)
    except (PipelineError,FileNotFoundError) as exc: print('ERROR: '+str(exc),flush=True); return 1
    return 0
if __name__=='__main__': raise SystemExit(main())
