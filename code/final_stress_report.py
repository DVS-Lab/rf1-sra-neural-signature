"""Result-driven reports; unavailable branches stay explicit, never fabricated."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import write_text,write_tsv

LIMITS=('Occipital exclusion is descriptive only: no model selection, no persistence success criterion, and no proof of abstract social coding. '
        'Phase-resolved SR outcome ↔ Trust outcome generalization is the primary stress test. '
        'It addresses a simple concurrent face/nonface explanation, while hemodynamic carryover, identity, familiarity, text, semantics and other perceptual information remain alternatives. '
        'UGR constant maps are visually confounded third-paradigm probes. '
        'Frozen DEV applications involve training-participant overlap; only participant-blocked CV endpoints test unseen development participants. '
        'The protected N=50 validation sample remains unscored.')

def table(frame,cols=None):
    if frame.empty: return 'Unavailable; see branch status and inventory.\n'
    f=frame[cols] if cols else frame
    def fmt(v): return f'{v:.4g}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join('---' for _ in f.columns)+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))+'\n'

def save(out,fig,name):
    fig.tight_layout()
    for ext in ('png','pdf','svg'): fig.savefig(out.output('results/figures/'+name+'.'+ext),dpi=250,bbox_inches='tight')
    plt.close(fig)

def phase_figure(out):
    fig,ax=plt.subplots(figsize=(12,5.5)); ax.axis('off')
    rows=[['Shared Reward decision','Partner picture + name; high/low guess','Explicit decision nuisance EVs'],
          ['Shared Reward outcome','Card value + valence symbol/color; no partner image/name','New phase-resolved outcome targets'],
          ['Trust decision','Partner picture + investment choices','Separate partner decision EVs'],
          ['Trust outcome','Partner text: shared/kept money + symbol/color; no face','Outcome targets; names/text remain'],
          ['UGR cue → offer/choice','Human face or computer image remains during offer','Model-3 constants include cue through choice endpoint'],
          ['UGR new categorical offer','Offer text/choice and face/computer cue','Offer onset → choice endpoint; pre-offer modeled separately']]
    t=ax.table(cellText=rows,colLabels=['Phase','Concurrent display (source-code audit)','Analysis role'],loc='center',cellLoc='left',colWidths=[.19,.47,.34]); t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1,3)
    ax.set_title('Figure 1. Task-phase audit: target timing and concurrent stimuli',pad=28)
    fig.text(.02,.01,'Source-code schematic, not screenshots. UGR remains visually confounded; no participant logs opened.',fontsize=9)
    save(out,fig,'Figure1_task_phases')

def matrix_figure(out,perf,family,number):
    sub=perf[(perf.group=='phase')&(perf.family==family)&~perf.model.str.contains('pooled')]
    domains=['sr_full','sr_outcome','trust']; a=np.full((3,3),np.nan); b=a.copy()
    for i in range(3):
        for j,d in enumerate(domains):
            for visual,mat in [(False,a),(True,b)]:
                r=sub[(sub.model==f'phase_{family}_{i}_'+('visual_excluded' if visual else 'whole_brain'))&(sub.test==d+'_'+family)]
                if len(r): mat[i,j]=r.accuracy.iloc[0]
    fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(np.ma.masked_invalid(a),cmap='RdBu_r',vmin=0,vmax=1)
    for i in range(3):
        for j in range(3): ax.text(j,i,f'{a[i,j]:.1%}\n○ {b[i,j]:.1%}' if np.isfinite(a[i,j]) else 'Unavailable',ha='center',va='center',fontsize=12)
    labels=['SR full-trial','SR outcome','Trust outcome']; ax.set(xticks=range(3),xticklabels=labels,yticks=range(3),yticklabels=labels,xlabel='Test: unseen development participants',ylabel='Training domain',title=f'Figure {number}. {family.replace("_"," ")}')
    fig.colorbar(im,ax=ax,label='Paired accuracy; chance=.50'); fig.text(.02,.01,'Cell: whole brain; ○ predefined occipital exclusion (descriptive only). Matched sample; fixed folds.',fontsize=9)
    save(out,fig,f'Figure{number}_'+family)

def profile_figure(out):
    path=out.output('work/partner_expression.tsv'); frame=pd.read_csv(path,sep='\t') if path.exists() else pd.DataFrame()
    fig,axes=plt.subplots(2,2,figsize=(11,8),squeeze=False)
    for i,family in enumerate(('social_context','friend_stranger_context')):
        for j,task in enumerate(('sr_full','trust')):
            ax=axes[i,j]
            if len(frame):
                g=frame[(frame.model==family)&(frame.task==task)]; p=g.pivot(index='subject',columns='partner',values='expression').reindex(columns=['computer','stranger','friend'])
                ax.plot(range(3),p.to_numpy().T,color='grey',alpha=.12,lw=.6)
                ax.plot(range(3),p.mean().to_numpy(),color='#b2182b',marker='o',lw=2)
            ax.set(xticks=range(3),xticklabels=['Computer','Stranger','Friend'],title=family.replace('_',' ')+' / '+task,ylabel='OOF expression (model-specific units)')
    fig.suptitle('Figure 4. Paired partner expression; each line is one development participant')
    save(out,fig,'Figure4_specificity')

def bars(out,perf,number,name,selection,note):
    g=perf[selection].copy(); fig,ax=plt.subplots(figsize=(11,max(4,.35*len(g)+1.8)))
    if len(g):
        y=np.arange(len(g)); x=g.accuracy.to_numpy(); lo=g.ci_low.to_numpy(); hi=g.ci_high.to_numpy()
        ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt='o',capsize=3); ax.set_yticks(y,[r.model+' → '+r.test+f' (N={r.n})' for r in g.itertuples()]); ax.invert_yaxis()
    else: ax.text(.5,.5,'Not available: see branch status / design gate',transform=ax.transAxes,ha='center')
    ax.axvline(.5,color='grey',ls='--'); ax.set(xlim=(0,1),xlabel='Paired accuracy with descriptive 95% interval',title=f'Figure {number}. {name}')
    fig.text(.01,.01,note,fontsize=9); save(out,fig,f'Figure{number}_'+('ugr_context' if number==5 else 'norm_violation'))

def endpoint_text(frame):
    if frame.empty: return 'not available (see inventory/design-gate status)'
    return '; '.join(f'{r.model} → {r.test}: {r.accuracy:.1%} (N={r.n})' for r in frame.itertuples())

def render(out,perf,perms,branch,keep,skipped):
    out.output('results/figures').mkdir(parents=True,exist_ok=True)
    phase_figure(out)
    for f,n in [('social_context',2),('friend_stranger_context',3)]: matrix_figure(out,perf,f,n)
    profile_figure(out)
    bars(out,perf,5,'UGR context: visually confounded third-task generalization',perf.test.isin(['ugr_context','ugr_high','ugr_low']),
         'Frozen DEV = training-participant overlap; other models = participant-blocked CV. Occipital exclusion descriptive only.')
    bars(out,perf,6,'Exploratory norm-violation transfer',perf.group.eq('norm'),
         'Positive = defection/unfair. Valence comparator sign reversed. Separate exploratory family; candidates unchanged.')
    keys=['family','group','test','evaluation']; visual=[]
    for r in perf[perf.visual.eq(True)].itertuples():
        match=perf[(perf.model==r.model.replace('_visual_excluded','_whole_brain'))&perf.test.eq(r.test)]
        if len(match): visual.append(dict(model=r.model,test=r.test,n=r.n,original_accuracy=float(match.accuracy.iloc[0]),visual_excluded_accuracy=r.accuracy,
            difference=r.accuracy-float(match.accuracy.iloc[0]),retained_voxels=int(keep.sum()),role='descriptive_only_not_selection_or_success_gate'))
    visual=pd.DataFrame(visual); write_tsv(out,'results/aggregate/visual_sensitivity.tsv',visual)
    cols=['model','test','n','accuracy','ci_low','ci_high','mean_margin','margin_ci_low','margin_ci_high','evaluation']
    phase=perf[(perf.group=='phase')&~perf.visual]; specific=perf[perf.group=='specificity']; fixed=perf[perf.group=='fixed_probe']; ugr=perf[perf.group=='ugr']; norm=perf[perf.group=='norm']
    questions=[('1. What was on screen?', 'See [TASK_PHASE_AUDIT.md](TASK_PHASE_AUDIT.md) and Figure 1. SR outcome: generic card/outcome display without partner name/face; Trust outcome: partner text and valence symbols without a face; UGR: face/computer cues remain during offers. Presentation code verifies the checked version, not every historical deployment.'),
      ('2–4. Outcome-only decoding and bidirectional context / friend–stranger transfer',table(phase,cols)+'\nThese matched-sample participant-blocked tests lead the scientific interpretation. Positive evidence would argue against simple concurrently displayed face/nonface differences across both target epochs; it would not eliminate carryover or semantic/identity alternatives.'),
      ('5. Predefined occipital exclusion',table(visual)+'\nDescriptive sensitivity only. Neither persistence nor failure determines candidate selection or success.'),
      ('6. Distinguishable context dimensions?',table(specific,cols)+'\nFigure 4 shows paired C/S/F expression. Above-chance cross-application is overlap, not independence. A nonsignificant F–S result does not establish F≈S; no equivalence threshold was specified. Compare profiles and both native/cross-application contrasts; margins across separately trained models are not on a common scale.'),
      ('7. Frozen social context → UGR',table(fixed,cols)+'\nThese are fixed-weight development probes with training-participant overlap, including phase-resolved SR probes. UGR is visually confounded; success alone cannot adjudicate perception.'),
      ('8. Outcome-trained context → UGR',table(ugr,cols)+'\nParticipant blocking is enforced. Training excludes concurrently displayed faces, but the UGR target remains visually confounded.'),
      ('9–11. Exploratory Trust ↔ UGR norm violation and generic valence',table(norm,cols)+'\nCategorical UGR status: '+branch['status']+'. '+str(branch.get('reason') or '')+'\n\nThe primary social model averages friend and stranger defection versus reciprocation. Computer conditions are separate. Generic valence uses the existing OOF model with score sign reversed so positive predicts violation; this is not a newly optimized valence model. Fairness amount/valence/choice can explain overlap; regional overlap or lack of regional correlations does not settle the distributed question.'),
      ('12. Collaborator summary and eventual preregistration','The two frozen candidates remain unchanged. These are final development diagnostics; none is an automatic validation gate. Review directional phase-transfer estimates and intervals, construct cross-application, UGR boundaries and the separately labeled exploratory norm family. Preserve the untouched N=50 and preregister endpoints, uncertainty/success rules and multiplicity before validation.')]
    body='# Final development stress tests\n\n'+LIMITS+'\n\n'
    body+='Current-source/QC subsets of the frozen N=192 development pool; no roster reassignment. Unknown QC never passes. Computational minimum is ten people and all five original folds, not a power guarantee. Original 95% coverage mask includes development CV test participants without labels and is retained; it is not fold-specific.\n\n'
    for title,text in questions: body+='## '+title+'\n\n'+text+'\n\n'
    body+='## Participant permutations\n\n'+table(perms)+'\n500 flips per primary new endpoint; a participant shares one flip across tasks in a family, including test labels; all five folds refit. Nominal development diagnostics, not project-wide multiplicity control.\n\n'
    body+='## Spatial comparisons\n\n'+table(pd.read_csv(out.output('results/aggregate/spatial_comparison.tsv'),sep='\t'))+'\nRaw weights are not regional importance; Haufe and mean patterns are descriptive.\n\n'
    body+='## Unavailable analyses\n\n'+(table(pd.DataFrame(skipped)) if skipped else 'None.\n')+'\nNo unavailable cell is encoded as zero or chance.\n\nholdout_scored = False\n'
    write_text(out,'reports/REPORT.md',body)
