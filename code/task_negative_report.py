"""One bounded empirical-control report and scientific comparison figure."""
import numpy as np
import pandas as pd
from utils import write_tsv,write_text
from specificity_design import DOMAINS,interval
from specificity_report import perm_stats
from task_negative_design import MODES,ENDPOINTS,REFERENCE

LABELS={'original':'Original outcome SVM','task_negative':'Task-negative one-feature','residualized':'Projection-removed SVM'}
NAMES=['SR → Trust: human–computer','Trust → SR: human–computer','SR → Trust: friend–stranger',
       'Trust → SR: friend–stranger','SR → Doors: social–monetary','Trust → Doors: social–monetary']
COLORS={'original':'#235789','task_negative':'#BD632F','residualized':'#39836C'}


def sanity_text(out):
    f=pd.read_csv(out.output('results/aggregate/template_sanity.tsv'),sep='\t'); folds=f[f.fold>0]
    s=pd.read_csv(out.output('results/aggregate/spatial_similarity.tsv'),sep='\t')
    cross=s[s.comparison=='cross_fold_task_negative'].spatial_r
    haufe=s[s.comparison=='task_negative_vs_outcome_social_context_haufe'].spatial_r
    warning=('**Limited negative support:** at least one training template has negative effects in <1% of mask voxels. Interpret predictive results cautiously; this descriptive flag does not select a different template. '
             if folds.limited_negative_support.any() else '')
    if not folds.dmn_negativity_enriched.all(): warning+='Negative effects are not more prevalent in DMN in every fold. This limits a specifically DMN-like interpretation; no atlas or threshold is changed. '
    return (warning+'\n\n' if warning else '')+f'''Decision COPE3 passes the modeled decision-versus-implicit-fixation audit. It is an operational estimate of task deactivation, not a resting-state or universal DMN signature. Source-code evidence alone cannot prove every historical deployment; the runtime audit additionally checks the retained runs' actual events, fitted EVs, contrasts, and geometry.

Across the five training templates: negative signed decision estimates cover {100*folds.negative_fraction.min():.1f}–{100*folds.negative_fraction.max():.1f}% of mask voxels; DMN {100*folds.dmn_negative_fraction.min():.1f}–{100*folds.dmn_negative_fraction.max():.1f}%; non-DMN {100*folds.non_dmn_negative_fraction.min():.1f}–{100*folds.non_dmn_negative_fraction.max():.1f}%. Mean negative-component magnitude per voxel is {folds.dmn_mean_negative_magnitude.min():.3g}–{folds.dmn_mean_negative_magnitude.max():.3g} in DMN and {folds.non_dmn_mean_negative_magnitude.min():.3g}–{folds.non_dmn_mean_negative_magnitude.max():.3g} outside it (original COPE units, not percent signal change). Cross-fold template r={cross.min():.3f}–{cross.max():.3f}; template versus phase-resolved human–computer Haufe r={haufe.min():.3f}–{haufe.max():.3f}. Cross-fold training sets overlap, so their similarity is not independent reliability. Spatial correlations are descriptive, with no spatial significance claim.'''


def figure(out,frame,differences,bg,n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nilearn import plotting
    fig=plt.figure(figsize=(17,10)); grid=fig.add_gridspec(3,2,width_ratios=[1.45,1],hspace=.35,wspace=.22)
    maps=[('signed_mean','A1. Signed decision − fixation (raw mean COPE; blue = negative)'),
          ('direction','A2. Negative component, then centered + unit norm (direction units)'),
          ('outcome_social_context_haufe','A3. Outcome human − computer Haufe (COPE / score units)')]
    displays=[]
    for i,(stem,title) in enumerate(maps):
        ax=fig.add_subplot(grid[i,0])
        d=plotting.plot_stat_map(str(out.output('results/maps/DEV_display_'+stem+'.nii.gz')),bg_img=bg['path'],
             axes=ax,figure=fig,display_mode='ortho',cut_coords=(0,-52,24),threshold=1e-10,
             symmetric_cbar=True,cmap='RdBu_r',colorbar=True,black_bg=False,annotate=True,draw_cross=False)
        ax.set_title(title,fontsize=10,loc='left',pad=14); displays.append(d)
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


def report(out,margins,expressions,null,bg):
    correct={m:(margins[m]>0).mean(2) for m in MODES}; n=len(correct['original'])
    maximum=abs(null-.5).max((1,2)); rows=[]; differences=[]; cells=[]
    for mi,mode in enumerate(MODES):
        for e,(a,b,_,_,role) in enumerate(ENDPOINTS):
            z=correct[mode][:,e]; lo,hi=interval(z,'task-negative-matched'); acc=float(z.mean())
            row=dict(mode=mode,endpoint=e,train=a,test=b,role=role,n=n,accuracy=acc,ci_low=lo,ci_high=hi)
            if e<2: row.update(perm_stats(acc,null[:,mi,e],maximum))
            rows.append(row)
            if mode!='original':
                delta=z-correct['original'][:,e]; dl,dh=interval(delta,'task-negative-matched')
                differences.append(dict(mode=mode,endpoint=e,train=a,test=b,n=n,accuracy_difference=float(delta.mean()),ci_low=dl,ci_high=dh))
            for v in range(4):
                values=(margins[mode][:,e,v]>0).astype(float); cl,ch=interval(values,'task-negative-matched')
                cells.append(dict(mode=mode,endpoint=e,train_valence=('positive','negative')[v//2],test_valence=('positive','negative')[v%2],n=n,accuracy=float(values.mean()),ci_low=cl,ci_high=ch))
    frame=pd.DataFrame(rows); diff=pd.DataFrame(differences)
    ex=pd.DataFrame(expressions); exrows=[]
    for domain in DOMAINS:
        g=ex[ex.domain==domain].sort_values('subject')
        for measure in ('first_score','second_score','signed_difference'):
            values=g[measure].to_numpy(); lo,hi=interval(values,'task-negative-matched')
            exrows.append(dict(domain=domain,measure=measure,n=len(values),mean=float(values.mean()),ci_low=lo,ci_high=hi))
    for name,data in [('transfers',frame),('paired_comparisons',diff),('raw_signed_expression',exrows),('valence_cells',cells)]:
        write_tsv(out,'results/aggregate/'+name+'.tsv',data)
    write_tsv(out,'results/aggregate/permutation_nulls.tsv',[dict(iteration=i,mode=m,endpoint=e,train=ENDPOINTS[e][0],test=ENDPOINTS[e][1],accuracy=float(null[i,j,e])) for i in range(len(null)) for j,m in enumerate(MODES) for e in range(2)])
    figure(out,frame,diff,bg,n)
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
          f'{len(null)} synchronized participant-label flips per variant, refitting each fold; the same flip applies to every task/valence and test label for that person. Existing original nulls are reused only after benchmark reconstruction and numerical reproduction of the first two null replicates. Templates remain fixed under permutations because construction is label-independent. MaxT uses all six primary endpoint × variant combinations as a complete-null familywise diagnostic, not a test of model differences. Null-centering flags (>0.03 absolute mean deviation): {int(frame[frame.role=="primary"].null_center_flag.sum())}/6. Full null quantiles, Monte Carlo tail intervals, unique values and centering diagnostics are in `transfers.tsv`; minimum p={1/(len(null)+1):.4f}. No secondary permutation matrix is launched.','',
          'Performance and matched-difference intervals use 10,000 participant-bootstrap resamples, keeping each person’s four valence cells together and identical resamples across variants. These fixed-prediction intervals omit uncertainty from refitting models and templates, and are descriptive. The original label-free development coverage mask includes CV test participants; it remains unchanged and is not training-fold-specific.','']
    # Present evidence numerically rather than choosing an outcome via new accuracy thresholds.
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
