"""One concise scientific report and comparison figure; no correspondence."""
import numpy as np
import pandas as pd
from scipy.stats import beta
from utils import write_text,write_tsv
from specificity_design import MODES,TASKS,DOMAINS,ENDPOINTS,CELLS,endpoint_correct,interval
from cross_valence_compute import summarize_cell

LABELS={'legacy':'Full-trial SR reference','outcome':'Outcome patterns','dmn_only':'DMN only','dmn_excluded':'DMN excluded','generic':'Generic task template'}
SHORT=('SR','Trust','Doors','SR (F−S)','Trust (F−S)')

def perm_stats(observed,null,maximum):
    count=len(null); k=int((null>=observed).sum()); deviation=abs(observed-.5)
    return dict(permutations=count,p_above_chance=(k+1)/(count+1),p_two_sided=(1+int((abs(null-.5)>=deviation).sum()))/(count+1),
        p_maxT_two_sided=(1+int((maximum>=deviation).sum()))/(count+1),null_mean=float(null.mean()),null_sd=float(null.std(ddof=1)),
        null_q025=float(np.quantile(null,.025)),null_q975=float(np.quantile(null,.975)),unique_null_values=len(np.unique(null)),
        null_mean_mc_se=float(null.std(ddof=1)/np.sqrt(count)),
        tail_probability_mc_low=float(beta.ppf(.025,k,count-k+1)) if k else 0.,
        tail_probability_mc_high=float(beta.ppf(.975,k+1,count-k)) if k<count else 1.,
        null_center_flag=bool(abs(null.mean()-.5)>.03))

def report(out,margins,null,overlap,mask_info):
    correct={m:endpoint_correct(margins[m]) for m in MODES}; rows=[]; matrix=[]; differences=[]
    maximum=abs(null-.5).max((1,2))
    for mi,mode in enumerate(MODES):
        for e,(a,b) in enumerate(ENDPOINTS):
            values=correct[mode][:,e]; lo,hi=interval(values,mode+str(e)); acc=float(values.mean())
            rows.append(dict(mode=mode,train=TASKS[a],test=TASKS[b],n=len(values),accuracy=acc,ci_low=lo,ci_high=hi,
                ci_method='paired participant bootstrap of fixed OOF correctness; four valence cells per person',
                **perm_stats(acc,null[:,mi,e],maximum)))
        for i,j in CELLS:
            a,b=DOMAINS[i],DOMAINS[j]
            matrix.append(dict(mode=mode,train=a,test=b,**summarize_cell(margins[mode][:,i,j],mode+a+b,10000)))
    for mode,reference in [('outcome','legacy'),('dmn_only','outcome'),('dmn_excluded','outcome'),('generic','outcome')]:
        for e,(a,b) in enumerate(ENDPOINTS):
            delta=correct[mode][:,e]-correct[reference][:,e]; lo,hi=interval(delta,mode+'-'+reference+str(e))
            differences.append(dict(mode=mode,reference=reference,train=TASKS[a],test=TASKS[b],n=len(delta),accuracy_difference=float(delta.mean()),ci_low=lo,ci_high=hi))
    frame=pd.DataFrame(rows); differences=pd.DataFrame(differences)
    write_tsv(out,'results/aggregate/central_transfers.tsv',frame); write_tsv(out,'results/aggregate/valence_matrix.tsv',matrix)
    write_tsv(out,'results/aggregate/paired_comparisons.tsv',differences); write_tsv(out,'results/aggregate/dmn_overlap.tsv',overlap)
    null_rows=[dict(iteration=i,mode=m,train=TASKS[a],test=TASKS[b],accuracy=float(null[i,mi,e])) for i in range(len(null)) for mi,m in enumerate(MODES) for e,(a,b) in enumerate(ENDPOINTS)]
    write_tsv(out,'results/aggregate/permutation_nulls.tsv',null_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(ax,bx)=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[2.2,1]})
    y=np.arange(len(ENDPOINTS)); colors=['#999999','#176B87','#9955A0','#2F8658','#D48C22']
    for i,mode in enumerate(MODES):
        g=frame[frame['mode']==mode]; yy=y+(i-2)*.13
        ax.errorbar(100*g.accuracy,yy,xerr=100*np.stack([g.accuracy-g.ci_low,g.ci_high-g.accuracy]),fmt='o',ms=4,capsize=2,label=LABELS[mode],color=colors[i])
    ax.axvline(50,color='black',ls=':',lw=1); ax.set_yticks(y,[SHORT[a]+' → '+SHORT[b] for a,b in ENDPOINTS]); ax.invert_yaxis()
    ax.set_xlim(0,103); ax.set_xlabel('Participant-averaged accuracy (%) with descriptive 95% CI')
    ax.set_title(f'A. Matched development transfers (N={len(next(iter(margins.values())))})'); ax.legend(fontsize=8,loc='lower left')
    ov=pd.DataFrame(overlap); kinds=['weights','haufe','mean_difference']; xx=np.arange(3)
    for j,f in enumerate(('social_context','friend_stranger_context')):
        g=ov[ov.family==f].set_index('pattern').loc[kinds]
        bx.bar(xx+(j-.5)*.32,g.absolute_mass_fraction*100,.32,label=f.replace('_',' '))
    bx.axhline(ov.dmn_voxel_fraction.iloc[0]*100,color='black',ls=':',label='DMN voxel fraction')
    bx.set_xticks(xx,['Decoder\nweights','Forward\npattern','Mean\ndifference']); bx.set_ylabel('Absolute spatial mass in DMN (%)'); bx.set_title('B. Frozen candidate overlap (N=192)'); bx.legend(fontsize=8)
    fig.suptitle('Development specificity: fixed atlas, fixed folds, no model selection',fontsize=14)
    fig.text(.5,.015,'DMN = independent Yeo-7 Default network; complement includes noncortical/unlabelled voxels. Holdout not accessed.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94])
    for ext in ('png','pdf'): fig.savefig(out.output('results/figures/specificity_comparison.'+ext),dpi=180)
    plt.close(fig)
    lines=['# Final development specificity','',f"N={len(next(iter(margins.values())))} for every model and comparison; original five folds, QC and 95% development-coverage mask retained. Validation remains unscored. No UGR/betrayal models executed.",'',
        '## Matched central results','', '| Transfer | Previous SR full-trial | Outcomes | DMN only | DMN excluded | Generic template |','|---|---|---|---|---|---|']
    for a,b in ENDPOINTS:
        cells=[]
        for mode in MODES:
            r=frame[(frame['mode']==mode)&(frame.train==TASKS[a])&(frame.test==TASKS[b])].iloc[0]
            cells.append(f'{100*r.accuracy:.1f}% [{100*r.ci_low:.1f}, {100*r.ci_high:.1f}]')
        lines.append('| '+SHORT[a]+' → '+SHORT[b]+' | '+' | '.join(cells)+' |')
    lines += ['', '| Outcome transfer | Change vs legacy (pp, 95% CI) | Permutation p (above chance) | Corrected two-sided p | Null mean ± SD |', '|---|---|---|---|---|']
    for a,b in ENDPOINTS:
        r=frame[(frame['mode']=='outcome')&(frame.train==TASKS[a])&(frame.test==TASKS[b])].iloc[0]
        d=differences[(differences['mode']=='outcome')&(differences.train==TASKS[a])&(differences.test==TASKS[b])].iloc[0]
        lines.append(f'| {SHORT[a]} → {SHORT[b]} | {100*d.accuracy_difference:+.1f} [{100*d.ci_low:+.1f}, {100*d.ci_high:+.1f}] | {r.p_above_chance:.4f} | {r.p_maxT_two_sided:.4f} | {r.null_mean:.3f} ± {r.null_sd:.3f} |')
    lines += ['', 'Accuracy is paired ordering: score(human)>score(computer), or score(friend)>score(stranger). Each transfer averages correctness across the four positive/negative training/testing cells within participant; all 52 individual cells per model are in `valence_matrix.tsv`. Doors→Doors retains same- and cross-valence positive controls. Trust and Doors inputs are unchanged; only SR switches from full-trial to outcome-only. Previous social-context OOF predictions must reproduce before any interpretation. F−S denotes friend–stranger transfers on the same matched N=178; its legacy comparison is a new matched-cohort refit, not the previous N=192 estimate.', '',
        '## Inference and scope','',f'{len(null)} synchronized participant label-flip permutations per variant refit every fold, including generic-template score orientation. Central-transfer diagnostics report above-chance and two-sided p values, max-|accuracy−0.5| correction across all 45 prespecified endpoint/variant combinations, null means/SD/quantiles, unique values, and Monte Carlo tail intervals. Max-statistic correction is a complete-null familywise diagnostic. Minimum p resolution is '+f'{1/(len(null)+1):.4f}. '+f'Null-centering flags: {int(frame.null_center_flag.sum())}/45 (descriptive absolute mean deviation >0.03; no rerun/selection rule).', '',
        'Paired participant-bootstrap intervals for outcome−legacy and each baseline−outcome are in `paired_comparisons.tsv`. These are descriptive intervals conditional on the fitted OOF predictions; they do not capture full retraining uncertainty. Cellwise intervals are exact binomial descriptive OOF intervals. No bootstrap unit is a map or fold. Permutation evidence for above-chance discrimination is not a test of differences between model variants.', '',
        '## Fixed references and controls','',f"Yeo 2011 seven-network liberal cortical atlas, label 7 (Default), nearest-neighbor resampled once into the frozen MNI152NLin6Asym grid. Analysis-mask intersection: {mask_info['dmn_voxels']:,}/{mask_info['mask_voxels']:,} voxels. This resting-state cortical DMN reference does not exhaust task-negative regions; DMN-excluded retains all other original-mask voxels, including unlabelled/subcortical voxels. No dilation, accuracy-selected threshold, mask search or C tuning.", '',
        'Generic task-response template: source-task training participants only, equal mean of six SR/Trust partner×valence COPEs (four social/monetary×valence Doors COPEs), spatially centered and unit norm. A fixed C=1 one-dimensional LinearSVC learns orientation from training projections; held-out participants contribute neither template nor orientation. This is an internal, label-blind direction control, not an external population task-negative template.', '',
        'All models retain independent per-map spatial centering (reapplied within restricted masks), without population feature scaling. The retained whole mask was defined using all development subjects, including subsequent CV test subjects; it is label-free but not training-fold-specific. Restricted masks therefore test different feature spaces, not equivalent model capacities.', '',
        'Frozen-candidate overlap characterizes the unchanged N=192 development fits, not held-out predictive performance. It uses original candidate masks and unchanged weights/forward patterns/mean differences: absolute and squared spatial-mass fractions, enrichment relative to DMN voxel fraction, signed spatial correlation with the binary reference. These threshold-free descriptive measures are not spatial significance tests. DMN overlap or survival outside DMN cannot establish an abstract social representation.', '',
        '## Remaining preregistration decisions','',
        '1. Confirm the primary construct and contrast (human–computer versus friend–stranger), and whether the already-frozen full-trial candidates remain primary or an outcome-only candidate needs a separately documented freeze. This package does not replace either candidate.',
        '2. Specify the external validation dataset, task/contrast mapping, QC and coverage rules, and primary endpoint/direction before accessing validation outcomes.',
        '3. Fix the confirmatory hypothesis family, multiplicity policy and success criterion; classify Doors and DMN checks as specificity evidence without choosing masks from these accuracies.',
        '4. Approve the protected N=50 scoring plan and execution timing. No protected images or scores were accessed here.', '',
        '[Comparison figure](../../../results/revised/specificity/figures/specificity_comparison.png). [Independent atlas documentation](https://freesurfer.net/fswiki/CorticalParcellation_Yeo2011).', '', 'Stop here: no additional exploratory analysis is launched.']
    write_text(out,'reports/SPECIFICITY.md','\n'.join(lines)+'\n')
    return frame
