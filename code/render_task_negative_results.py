"""Reporting-only recovery of the a500929 plotting failure; no private inputs or fits.

Keep the original calculation code/identity untouched. Only committed aggregates,
development group maps and a standard anatomical background are read.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from utils import load_config,sha256,write_json,write_text,PipelineError
from characterization_audit import require
from task_negative_design import output_config,MODES,ENDPOINTS,REFERENCE
from task_negative_report import LABELS,NAMES,COLORS,sanity_text


def orthogonal_slices(image,background):
    image=nib.as_closest_canonical(image)
    affine=image.affine; spacing=np.diag(affine)[:3]
    require(np.all(spacing>0) and np.allclose(affine[:3,:3],np.diag(spacing),atol=1e-5),'display requires axis-aligned MNI grid')
    values=image.get_fdata(); anatomy=resample_from_to(background,(image.shape,affine),order=1).get_fdata()
    require(values.ndim==3 and np.isfinite(values).all() and np.isfinite(anatomy).all(),'nonfinite group display or standard anatomy')
    indices=np.rint(nib.affines.apply_affine(np.linalg.inv(affine),[0,-52,24])).astype(int)
    require(np.all(indices>=0) and np.all(indices<image.shape),'fixed display cuts outside group map')
    planes=[]
    for axis in (1,0,2):
        other=[a for a in range(3) if a!=axis]
        extent=[affine[a,3]+spacing[a]*v for a in other for v in (-.5,image.shape[a]-.5)]
        coord=affine[axis,3]+spacing[axis]*indices[axis]
        planes.append((np.take(values,indices[axis],axis=axis).T,np.take(anatomy,indices[axis],axis=axis).T,
                       extent,f'{"xyz"[axis]}={coord:.1f}',axis))
    return planes,float(np.max(abs(values))),float(np.max(anatomy))


def figure(out,frame,differences,bg,n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(17,10)); grid=fig.add_gridspec(3,2,width_ratios=[1.45,1],hspace=.35,wspace=.22)
    maps=[('signed_mean','A1. Signed decision − fixation (raw mean COPE; blue = negative)'),
          ('direction','A2. Negative component, then centered + unit norm (direction units)'),
          ('outcome_social_context_haufe','A3. Outcome human − computer Haufe (COPE / score units)')]
    anatomy=nib.load(bg['path'])
    for i,(stem,title) in enumerate(maps):
        row=grid[i,0].subgridspec(1,4,width_ratios=[1,1,1,.055],wspace=.04)
        image=nib.load(out.output('results/maps/DEV_display_'+stem+'.nii.gz'))
        planes,limit,anatomy_limit=orthogonal_slices(image,anatomy)
        require(limit>0 and anatomy_limit>0,'empty display map/background')
        for j,(values,background,extent,coordinate,axis) in enumerate(planes):
            ax=fig.add_subplot(row[0,j]); ax.set_facecolor('white')
            ax.imshow(np.ma.masked_less_equal(background,0),origin='lower',extent=extent,cmap='gray',vmin=0,vmax=anatomy_limit)
            overlay=ax.imshow(np.ma.masked_where(abs(values)<=1e-10,values),origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
            ax.text(.02,.02,coordinate,transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
            if axis!=0:
                for pos,label in ((.04,'L'),(.90,'R')): ax.text(pos,.90,label,transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
            if j==0: ax.text(0,1.10,title,transform=ax.transAxes,fontsize=10,ha='left')
        fig.colorbar(overlay,cax=fig.add_subplot(row[0,3]),format='%.2g')
    ax=fig.add_subplot(grid[:2,1]); y=np.arange(2)
    for mi,mode in enumerate(MODES):
        z=frame[(frame['mode']==mode)&(frame.role=='primary')].sort_values('endpoint')
        ax.errorbar(100*z.accuracy,y+(mi-1)*.16,xerr=100*np.stack([z.accuracy-z.ci_low,z.ci_high-z.accuracy]),
                    fmt='o',ms=6,capsize=3,color=COLORS[mode],label=LABELS[mode])
    ax.axvline(50,color='black',lw=1,ls=':'); ax.set_xlim(0,101); ax.set_ylim(1.55,-.55)
    ax.set_yticks(y,['SR → Trust','Trust → SR']); ax.set_xlabel('Paired-ordering accuracy (%) • 95% participant CI')
    ax.set_title(f'B. Human–computer transfers • matched N={n}',loc='left',fontsize=12)
    ax.legend(loc='lower left',fontsize=9); ax.spines[['top','right']].set_visible(False)
    dx=fig.add_subplot(grid[2,1])
    for mi,mode in enumerate(MODES[1:]):
        z=differences[(differences['mode']==mode)&(differences.endpoint<2)].sort_values('endpoint')
        dx.errorbar(100*z.accuracy_difference,y+(mi-.5)*.18,
                    xerr=100*np.stack([z.accuracy_difference-z.ci_low,z.ci_high-z.accuracy_difference]),
                    fmt='o',ms=5,capsize=3,color=COLORS[mode])
    dx.axvline(0,color='black',ls=':',lw=1); dx.set_yticks(y,['SR → Trust','Trust → SR']); dx.set_ylim(1.5,-.5)
    dx.set_xlabel('Change versus original (percentage points)'); dx.set_title('Matched differences • 95% participant CI',loc='left',fontsize=11)
    dx.spines[['top','right']].set_visible(False)
    fig.suptitle('Empirical task-negative control • development only',fontsize=16,y=.98)
    fig.text(.5,.025,'Maps: full-development displays only; separate scales/units. All prediction templates use fold-training people only.\nFixed-prediction intervals omit refitting uncertainty. Protected validation remains unscored.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.91,bottom=.13,left=.025,right=.98)
    for ext in ('png','pdf'): fig.savefig(out.output('results/figures/task_negative_comparison.'+ext),dpi=180)
    plt.close(fig)


def summary(out,frame,diff,exrows,count,n):
    text=['# Empirical task-negative control','',f'Development only: identical N={n}, original five participant folds, QC and whole-brain mask for every comparison. Protected N=50 remains unscored. Reference: `{REFERENCE}`. All prior outputs and frozen candidates are preserved.','',
          '## Source and spatial checks','',sanity_text(out),'',
          'The source template specifies EVs win, loss, decision and missed decision; COPE3 weights are [0, 0, 1, 0]. The task presentation draws fixation between decisions and feedback and during jittered ITIs (nominal 1.1–11.6 s). All retained runs must contain 40 decision/feedback pairs, positive interior fixation gaps, matching canonical and fitted EV timings, and an estimable decision contrast. The audit does not assume the initial/final unused acquisition time is rest. Missed-decision timing is a nuisance EV; the task code can log scheduled feedback labels even when showing a missed-response message. No new exclusion is introduced. Aggregate actual baseline timing and missed-decision counts are in `baseline_audit.tsv`.','',
          'Each training-only raw signed mean is clipped at zero *before* spatial centering and L2 normalization. Source COPEs are never precentered or sign-inverted. Negative magnitude is summarized per voxel in DMN and its complement, including zero-valued negative components. The existing Yeo-7 Default label is anatomical context only; no threshold or atlas is selected from performance.','',
          'The Haufe comparator is a descriptive fixed-C human–computer fit pooling positive/negative outcome maps from Shared Reward and Trust on each same fold-training set. It is not a new frozen candidate. Full-N display maps are separately labelled `DEV_display`; none enters CV. Signed response COPE, centered unit direction, and Haufe covariance/score pattern have distinct units. Positive voxels in the centered direction need not represent positive task-versus-baseline responses.','',
          '## Matched transfers','', '| Transfer | Original | Task-negative expression | Projection removed |','|---|---|---|---|']
    for e,name in enumerate(NAMES):
        entries=[]
        for m in MODES:
            r=frame[(frame['mode']==m)&(frame.endpoint==e)].iloc[0]
            entries.append(f'{100*r.accuracy:.1f}% [{100*r.ci_low:.1f}, {100*r.ci_high:.1f}]')
        text.append('| '+name+(' (primary)' if e<2 else ' (secondary)')+' | '+' | '.join(entries)+' |')
    text+=['','| Primary comparison versus original | Change (pp; 95% CI) | Above-chance permutation p | Two-sided maxT p | Null mean ± SD |','|---|---|---|---|---|']
    for e in range(2):
        for m in MODES:
            r=frame[(frame['mode']==m)&(frame.endpoint==e)].iloc[0]
            d=diff[(diff['mode']==m)&(diff.endpoint==e)]
            change='reference' if d.empty else f'{100*d.iloc[0].accuracy_difference:+.1f} [{100*d.iloc[0].ci_low:+.1f}, {100*d.iloc[0].ci_high:+.1f}]'
            text.append(f'| {NAMES[e]} / {LABELS[m]} | {change} | {r.p_above_chance:.4f} | {r.p_maxT_two_sided:.4f} | {r.null_mean:.3f} ± {r.null_sd:.3f} |')
    text+=['','Accuracy is paired ordering, averaging correctness across four train/test-valence cells within each person. Original outcome-only margins must reproduce the completed specificity run (tolerance 1e-5 absolute / 2e-5 relative, identical correctness). Source features, centering, C=1, labels, orientations and voxel-model dual solver are unchanged. The independent spatial template uses no social labels; its one-dimensional C=1 readout uses training labels and the fixed primal solver, as in the prior one-feature baseline. Residualization removes exactly one centered unit direction from training and test maps before refitting the original SVM.','',
          '## Raw expression direction','', '| Outcome domain | First − second mean [95% CI] |','|---|---|']
    for r in exrows:
        if r['measure']=='signed_difference': text.append(f"| {r['domain']} | {r['mean']:+.3g} [{r['ci_low']:+.3g}, {r['ci_high']:+.3g}] |")
    text+=['','First/second means human/computer, friend/stranger, or social/monetary Doors, respectively. Positive differences mean greater alignment with the centered task-negative direction in the first condition; they are signed dot products, not absolute deactivation relative to fixation in that social condition. Positive/negative valences remain separate; both condition-score means and intervals are in `raw_signed_expression.tsv`.','',
          '## Inference and interpretation','',
          f'{count} synchronized participant-label flips per variant, refitting each fold; the same flip applies to every task/valence and test label for that person. Existing original nulls are reused only after benchmark reconstruction and numerical reproduction of the first two null replicates. Templates remain fixed under permutations because construction is label-independent. MaxT uses all six primary endpoint × variant combinations as a complete-null familywise diagnostic, not a test of model differences. Null-centering flags (>0.03 absolute mean deviation): {int(frame[frame.role=="primary"].null_center_flag.sum())}/6. Full null quantiles, Monte Carlo tail intervals, unique values and centering diagnostics are in `transfers.tsv`; minimum p={1/(count+1):.4f}. No secondary permutation matrix is launched.','',
          'Performance and matched-difference intervals use 10,000 participant-bootstrap resamples, keeping each person’s four valence cells together and identical resamples across variants. These fixed-prediction intervals omit uncertainty from refitting models and templates, and are descriptive. The original label-free development coverage mask includes CV test participants; it remains unchanged and is not training-fold-specific.','']
    # Existing estimates only; no refitting, resampling or model selection.
    for e in range(2):
        t=frame[(frame['mode']=='task_negative')&(frame.endpoint==e)].iloc[0]
        r=frame[(frame['mode']=='residualized')&(frame.endpoint==e)].iloc[0]
        d=diff[(diff['mode']=='residualized')&(diff.endpoint==e)].iloc[0]
        text.append(f'{NAMES[e]}: expression alone achieves {100*t.accuracy:.1f}%; residual decoding achieves {100*r.accuracy:.1f}% (change {100*d.accuracy_difference:+.1f} pp, CI {100*d.ci_low:+.1f} to {100*d.ci_high:+.1f}).')
    text+=['','Interpret these estimates and intervals against three possibilities: **A**, strong expression prediction plus a substantial residual drop supports a generic deactivation explanation; **B**, weak expression prediction with retained residual decoding supports information beyond this direction; **C**, expression prediction alongside retained residual decoding supports partial overlap. No new cutoff is imposed to force an A/B/C label, and a nonsignificant difference does not establish equivalence. A one-direction projection cannot remove all generic task signals. DMN anatomy and task-negative function are not synonymous; neither survival nor anatomical location proves abstract social coding.','',
          'Social Doors transfers are secondary: training-only monetary Doors decisions supply the template while monetary Doors outcomes enter the target comparison. People remain separated and phases are modeled separately, but these endpoints share task context and are not independent template validation.','',
          '## Decisions for preregistration — stop development here','',
          '1. Agree whether the claim is human–computer social context, friend–stranger closeness, generic deactivation, or overlapping information; state the limits of the task manipulations.',
          '2. Choose among the already documented candidate strategies and fix the final source phase, contrast, training sample and scoring method. This control does not replace the frozen candidates or select a residualized/atlas-restricted model.',
          '3. Fix the validation datasets, contrast mapping, QC/coverage rules, primary endpoint/direction, multiplicity correction and success criterion before accessing validation outcomes.',
          '4. Preregister the protected N=50 execution plan and external validation. No UGR/betrayal, additional template, atlas, threshold or parameter search follows this report.','',
          '[Comparison figure](../../../results/revised/task_negative_control/figures/task_negative_comparison.png). All statistics are development results.']
    write_text(out,'reports/REPORT.md','\n'.join(text)+'\n')
    return frame


def validated_tables(out):
    read=lambda name:pd.read_csv(out.output('results/aggregate/'+name+'.tsv'),sep='\t')
    frame=read('transfers'); differences=read('paired_comparisons'); expression=read('raw_signed_expression'); null=read('permutation_nulls')
    require(len(frame)==18 and not frame.duplicated(['mode','endpoint']).any() and frame.n.eq(178).all(),'incomplete matched transfers')
    require(set(zip(frame['mode'],frame.endpoint))=={(m,e) for m in MODES for e in range(6)},'unexpected endpoints')
    require(len(differences)==12 and not differences.duplicated(['mode','endpoint']).any() and differences.n.eq(178).all(),'incomplete paired comparisons')
    require(len(expression)==30 and not expression.duplicated(['domain','measure']).any() and expression.n.eq(178).all(),'incomplete expressions')
    require(len(null)==3000 and not null.duplicated(['iteration','mode','endpoint']).any(),'incomplete indexed nulls')
    require(set(zip(null['mode'],null.endpoint))=={(m,e) for m in MODES for e in range(2)},'unexpected null endpoints')
    for (_,e),g in null.groupby(['mode','endpoint']): require(np.array_equal(np.sort(g.iteration),np.arange(500)),'missing null replicate')
    for data,columns in ((frame,['accuracy','ci_low','ci_high']), (differences,['accuracy_difference','ci_low','ci_high']), (expression,['mean','ci_low','ci_high'])):
        require(np.isfinite(data[columns].to_numpy()).all(),'nonfinite aggregate')
    require(frame.accuracy.between(0,1).all() and null.accuracy.between(0,1).all(),'invalid accuracy')
    # Independently verify the published null diagnostics, without new permutations.
    from specificity_report import perm_stats
    maximum=null.assign(deviation=abs(null.accuracy-.5)).groupby('iteration').deviation.max().to_numpy()
    for r in frame[frame.role=='primary'].itertuples():
        values=null[(null['mode']==r.mode)&(null.endpoint==r.endpoint)].sort_values('iteration').accuracy.to_numpy()
        expected=perm_stats(r.accuracy,values,maximum)
        for k,v in expected.items(): require(np.isclose(float(getattr(r,k)),float(v),atol=1e-12,rtol=0),'published permutation diagnostic mismatch: '+k)
    for r in differences.itertuples():
        a=frame[(frame['mode']==r.mode)&(frame.endpoint==r.endpoint)].accuracy.iloc[0]
        b=frame[(frame['mode']=='original')&(frame.endpoint==r.endpoint)].accuracy.iloc[0]
        require(np.isclose(a-b,r.accuracy_difference,atol=1e-12,rtol=0),'published paired difference mismatch')
    for name,count in (('benchmark_reconstruction',6),('original_null_reconstruction',2)):
        values=read(name); require(len(values)==count and (values.max_absolute_error<=1e-5).all(),'original reconstruction check failed')
    return frame,differences,expression


def run(base,background):
    out=output_config(base); contract=json.loads((base.root/'config/task_negative_render.json').read_text())
    # The original failure and numerical results are preserved as an immutable source record.
    for rel,h in contract['inputs'].items(): require(sha256(base.root/rel)==h,'report recovery input changed: '+rel)
    spec=json.loads((base.root/'config/task_negative_control.json').read_text())
    for rel,h in spec['frozen'].items(): require(sha256(base.root/rel)==h,'pre-existing dependency changed: '+rel)
    state=json.loads(out.output('provenance/run_status.json').read_text())
    require(state['status']=='failed' and state['stage']=='report and figure' and state['error_type']=='ImportError' and not state['holdout_scored'] and state['original_outputs_unchanged'],'unrecognized recovery state')
    progress=json.loads(out.output('provenance/permutation_progress.json').read_text())
    require(progress['completed']==progress['total']==1000 and not progress['holdout_scored'],'permutations incomplete')
    audit=json.loads(out.output('provenance/baseline_audit.json').read_text())
    require(audit['status']=='passed' and audit['n']==178 and not audit['holdout_scored'],'source audit incomplete')
    background=Path(background).resolve()
    require(not any(p.startswith('sub-') for p in background.parts),'participant image is not a standard background')
    require(sha256(background)==contract['background_sha256'],'standard background differs from reviewed display asset')
    frame,differences,expression=validated_tables(out)
    out.output('results/figures').mkdir(parents=True,exist_ok=True)
    bg=dict(path=str(background),sha256=sha256(background))
    figure(out,frame,differences,bg,178)
    summary(out,frame,differences,expression.to_dict('records'),500,178)
    report=out.output('reports/REPORT.md'); text=report.read_text()
    text=text.replace('## Decisions for preregistration — stop development here',
        'The observed pattern supports **partial overlap**: the independent task-negative direction strongly predicts SR→Trust, while both transfers retain substantial decoding after its removal. This does not show equivalence of the models or exclude other generic task signals.\n\n## Decisions for preregistration — stop development here')
    text+='\nReporting recovery: all 1,000 new permutation jobs completed on Linux2 before a Nilearn/Matplotlib import incompatibility stopped plotting. This report and figure were rendered from the hash-verified committed aggregates and development group maps; no participant images, model fits, bootstrap resamples or new permutations were used. The original failed calculation status is retained for provenance. `render_status.json` records completion of this reporting-only recovery.\n'
    write_text(out,'reports/REPORT.md',text)
    for rel,h in contract['inputs'].items(): require(sha256(base.root/rel)==h,'report rendering altered a calculation input')
    outputs=['reports/REPORT.md','results/figures/task_negative_comparison.png','results/figures/task_negative_comparison.pdf']
    write_json(out,'provenance/render_status.json',dict(status='complete',scope='reporting_only',source_commit=contract['source_commit'],
        original_calculation_status_preserved=True,calculation_inputs_unchanged=True,holdout_scored=False,
        private_images_accessed=False,models_refit=False,permutations_rerun=False,n=178,background_sha256=bg['sha256'],
        code_sha256=sha256(Path(__file__)),contract_sha256=sha256(base.root/'config/task_negative_render.json'),
        software={p:importlib.metadata.version(p) for p in ('numpy','scipy','nibabel','pandas','matplotlib')},
        products={str(out.output(p).relative_to(base.root)):sha256(out.output(p)) for p in outputs}))
    print(report)
    print('Reporting complete. Numerical results and calculation checkpoints unchanged. No analyses rerun.')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--background',required=True,type=Path); a=p.parse_args()
    try: run(load_config(),a.background)
    except (PipelineError,FileNotFoundError) as exc: print('ERROR: '+str(exc)); return 1
    return 0
if __name__=='__main__': raise SystemExit(main())
