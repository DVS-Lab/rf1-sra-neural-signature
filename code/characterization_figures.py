"""Scientific figures: fixed contrasts; descriptive thresholds; no automatic downloads."""
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr, spearmanr
from characterization_audit import TASKS, TARGETS, PRIMARY, SENSITIVITY
from characterization_compute import correlation
from reporting import icc31
from utils import atomic_output, write_tsv

LABELS={'sharedreward':'Shared Reward','trust':'Trust','socialdoors':'Social / monetary Doors'}
FAMILY={'social_context':'Social context','social_reward':'Social reward modulation',
        'closeness_positive':'Closeness: positive outcome','closeness_negative':'Closeness: negative outcome',
        'closeness_reward':'Closeness: reward modulation','valence':'Outcome valence'}
DIRECTION={'social_context':('SOCIAL','NONSOCIAL'),'social_reward':('SOCIAL','NONSOCIAL'),
           'closeness_positive':('FRIEND','STRANGER'),'closeness_negative':('FRIEND','STRANGER'),
           'closeness_reward':('FRIEND','STRANGER'),'valence':('POSITIVE','NEGATIVE')}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})


def save(out, name, fig, vector=True):
    for suffix in ('png','pdf','svg') if vector else ('png',):
        with atomic_output(out,'results/figures/'+name+'.'+suffix) as p:
            fig.savefig(p,dpi=300,bbox_inches='tight')
    plt.close(fig)


def canonical(volume, reference):
    image=nib.as_closest_canonical(nib.Nifti1Image(np.asarray(volume,np.float32),reference.affine))
    return image.get_fdata(),image.affine


def one_slice(ax, volume, mask, reference, title):
    data,aff=canonical(volume,reference); support,_=canonical(mask,reference)
    indices=np.flatnonzero(support.any(axis=(0,1)))
    z=int(indices[len(indices)//2])
    vmax=float(np.max(np.abs(data))) or 1.
    ax.imshow(support[:,:,z].T,origin='lower',cmap='Greys',vmin=0,vmax=2)
    im=ax.imshow(np.ma.masked_where((support[:,:,z]==0)|(data[:,:,z]==0),data[:,:,z]).T,
              origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    ax.set_title(title+f'\nz={aff[2,2]*z+aff[2,3]:.0f} mm; L ← → R',fontsize=9)
    ax.axis('off')
    ax.figure.colorbar(im,ax=ax,shrink=.6,fraction=.04,pad=.02)


def brain_figures(out,stem,volume,mask,reference,family,kind,spec):
    data,aff=canonical(volume,reference); support,_=canonical(mask,reference)
    values=data[support>0]; vmax=float(np.max(np.abs(values))) or 1.
    pos,neg=DIRECTION[family]
    direction=f'Positive → {pos}; negative → {neg}'
    for percentile in (0,95,99):
        threshold=float(np.percentile(np.abs(values),percentile)) if percentile else 0.
        shown=np.where(np.abs(data)>=threshold,data,0)
        fig,axes=plt.subplots(3,5,figsize=(10,6.5))
        for axis,label in enumerate(('sagittal x','coronal y','axial z')):
            occupied=np.flatnonzero((support>0).any(axis=tuple(a for a in range(3) if a!=axis)))
            cuts=np.linspace(occupied[0],occupied[-1],7)[1:-1].astype(int)
            for ax,cut in zip(axes[axis],cuts):
                sl=np.take(shown,cut,axis=axis).T; bg=np.take(support,cut,axis=axis).T
                plane=[a for a in range(3) if a!=axis]
                aspect=abs(aff[plane[1],plane[1]]/aff[plane[0],plane[0]])
                ax.imshow(bg,origin='lower',cmap='Greys',vmin=0,vmax=2,aspect=aspect)
                im=ax.imshow(np.ma.masked_where((bg==0)|(sl==0),sl),origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect=aspect)
                ax.set_title(f'{label}={aff[axis,axis]*cut+aff[axis,3]:.0f} mm',fontsize=8); ax.axis('off')
        view='unthresholded' if not percentile else f'top {100-percentile}% absolute values — display only'
        fig.suptitle(f'{FAMILY[family]} | {kind}\n{view}\n{direction}; RAS orientation (L→R in coronal/axial)',fontsize=10)
        fig.subplots_adjust(top=.80,right=.91,hspace=.32)
        fig.colorbar(im,cax=fig.add_axes([.93,.2,.015,.5]),label=kind)
        suffix='unthresholded' if not percentile else f'top{100-percentile}pct'
        save(out,stem+'_'+suffix,fig,vector=False)
        if spec.get('glass_brain',True):
            try:
                from nilearn.plotting import plot_glass_brain
            except ImportError: continue
            fig=plt.figure(figsize=(9,3.5))
            plot_glass_brain(nib.Nifti1Image(shown.astype(np.float32),aff),display_mode='lyrz',
                            plot_abs=False,colorbar=True,cmap='RdBu_r',symmetric_cbar=True,vmax=vmax,
                            threshold=None,figure=fig,title=f'{kind}: {view}; {direction}')
            save(out,stem+'_glass_'+suffix,fig,vector=False)
    fig,ax=plt.subplots(figsize=(6,3.5)); ax.hist(values,bins=80,color='#476b85')
    ax.axvline(0,color='black',lw=.8); ax.set(xlabel=kind,ylabel='Voxels',title=FAMILY[family]+' — '+direction)
    fig.tight_layout(); save(out,stem+'_histogram',fig)


def heatmap(ax, matrix, names, title, performance=False, n=None):
    norm=TwoSlopeNorm(vmin=0,vcenter=.5,vmax=1) if performance else TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
    im=ax.imshow(matrix,cmap='RdBu_r',norm=norm)
    ax.set_xticks(range(len(names)),names,rotation=30,ha='right'); ax.set_yticks(range(len(names)),names)
    ax.set_title(title+(f'\nN={n}; chance=.50' if performance else ''))
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j,i,f'{matrix[i,j]:.1%}' if performance else f'{matrix[i,j]:.2f}',ha='center',va='center',
                    color='white' if (abs(matrix[i,j]-(.5 if performance else 0))>(.32 if performance else .7)) else 'black',fontsize=9)
    return im


def point_range(ax, rows, labels, title):
    acc=rows.accuracy.to_numpy(float); lo=rows.ci_low.to_numpy(float); hi=rows.ci_high.to_numpy(float)
    y=np.arange(len(rows))
    ax.errorbar(acc,y,xerr=[acc-lo,hi-acc],fmt='o',color='#216b84',capsize=3)
    ax.axvline(.5,color='gray',ls='--'); ax.set_yticks(y,labels); ax.invert_yaxis()
    ax.set(xlim=(0,1),xlabel='OOF forced-choice accuracy (95% CI)',title=title); ax.grid(axis='x',alpha=.15)


def fixed_figures(out,scopes):
    fig,ax=plt.subplots(figsize=(12,6)); ax.axis('off')
    boxes=[(.04,.66,.26,.24,'TASKS\nShared Reward (full trial)\nTrust (outcome)\nSocial / monetary Doors (outcome)'),
           (.37,.60,.27,.35,'FROZEN CONTRAST FAMILIES\nContext: average across valence\nReward: positive − negative\nCloseness: friend − stranger\nValence: positive − negative'),
           (.71,.64,.25,.28,'MODELS\nTask-specific\nCommon pooled\nLeave one paradigm out'),
           (.06,.12,.88,.25,'Five participant-blocked folds; all tasks from each test person excluded from training\nPrimary development: partner pair N='+str(len(scopes[(PRIMARY,'partner_pair')]['subjects']))+'; three paradigms N='+str(len(scopes[(PRIMARY,'three_paradigm')]['subjects']))+'\nFrozen development-only masks and LinearSVC settings; validation N=50 remains untouched')]
    from matplotlib.patches import FancyBboxPatch
    for x,y,w,h,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor='#edf3f6',edgecolor='#355b70'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    for a,b in [((.31,.77),(.35,.77)),((.65,.77),(.69,.77)),((.83,.59),(.83,.40))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=2))
    ax.set_title('Figure 1. Frozen v4 development analysis and characterization',fontsize=14); save(out,'Figure1_schematic',fig)
    perf=scopes[(PRIMARY,'three_paradigm')]['performance']; tasks=TASKS['three_paradigm']
    fig,axes=plt.subplots(2,2,figsize=(13,10),gridspec_kw={'height_ratios':[1.2,1]})
    for j,family in enumerate(TARGETS['three_paradigm']):
        rows=perf[perf.family==family]; pair=rows[rows.scope=='pairwise']
        mat=pair.pivot(index='train_tasks',columns='test_task',values='accuracy').reindex(index=tasks,columns=tasks).to_numpy(float)
        im=heatmap(axes[0,j],mat,[LABELS[t] for t in tasks],FAMILY[family],True,int(rows.n.iloc[0]))
        axes[0,j].set(xlabel='Test paradigm',ylabel='Training paradigm'); fig.colorbar(im,ax=axes[0,j],shrink=.7)
        lopo=rows[rows.scope=='lopo'].set_index('test_task').loc[list(tasks)]
        point_range(axes[1,j],lopo,['Held out: '+LABELS[t] for t in tasks],'Leave-one-paradigm-out')
    fig.suptitle('Figure 2. Social context versus social reward modulation',fontsize=14); fig.tight_layout(); save(out,'Figure2_context_reward_transfer',fig)


def similarity_figures(out, patterns):
    rows=[]
    for family in TARGETS['three_paradigm']:
        names=[*TASKS['three_paradigm'],'common']
        for kind,suffix in [('weight','weight'),('haufe','haufe'),('mean_difference','meanmap')]:
            vectors=[patterns[('three_paradigm',family,n)][kind] for n in names]
            matrix=np.array([[correlation(a,b) for b in vectors] for a in vectors])
            rows.extend({'family':family,'kind':kind,'model_a':a,'model_b':b,'spatial_r':matrix[i,j]}
                        for i,a in enumerate(names) for j,b in enumerate(names))
            fig,ax=plt.subplots(figsize=(6,5)); im=heatmap(ax,matrix,[LABELS.get(n,n.title()) for n in names],FAMILY[family]+' — '+kind)
            fig.colorbar(im,ax=ax,label='Spatial Pearson r'); fig.tight_layout(); save(out,family+'_'+suffix+'_similarity',fig)
    write_tsv(out,'results/aggregate/task_pattern_similarity.tsv',rows)
    return pd.DataFrame(rows)


def main_pattern_figures(out,scopes,patterns,stability,geometries):
    cohort='three_paradigm'; family='social_context'; mask,ref=geometries[cohort]
    common=patterns[(cohort,family,'common')]
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    from build_mask import reconstruct
    for ax,values,title in zip(axes[0],[common['weight'],common['haufe'],stability[(cohort,family)]['haufe']['sign_stable']],
                               ['Raw discriminative weights','Descriptive Haufe pattern','Bootstrap Haufe sign-stable (display only)']):
        one_slice(ax,reconstruct(values,mask),mask,ref,title)
    names=[*TASKS[cohort],'common']
    for ax,kind in zip(axes[1],['weight','haufe','mean_difference']):
        vectors=[patterns[(cohort,family,n)][kind] for n in names]
        matrix=np.array([[correlation(a,b) for b in vectors] for a in vectors])
        im=heatmap(ax,matrix,['SR','Trust','Doors','Common'],kind+' spatial similarity'); fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('Figure 3. Common social-context pattern\nPositive → SOCIAL; negative → NONSOCIAL. Descriptive, not inferential.',fontsize=13)
    fig.tight_layout(); save(out,'Figure3_social_context_pattern',fig)
    fig=plt.figure(figsize=(17,9)); grid=fig.add_gridspec(2,4,height_ratios=[1.4,1])
    cohort='partner_pair'; perf=scopes[(PRIMARY,cohort)]['performance']; mask,ref=geometries[cohort]
    for j,family in enumerate(('closeness_positive','closeness_negative','closeness_reward','valence')):
        rows=perf[(perf.family==family)&perf.scope.isin(['pairwise','common'])].copy()
        labels=[('Common' if r.scope=='common' else ('SR' if r.train_tasks=='sharedreward' else 'Trust'))+' → '+('SR' if r.test_task=='sharedreward' else 'Trust') for r in rows.itertuples()]
        ax=fig.add_subplot(grid[0,j]); point_range(ax,rows,labels,FAMILY[family])
        ax=fig.add_subplot(grid[1,j])
        if family=='closeness_reward':
            ax.axis('off'); ax.text(.1,.5,'Reward modulation:\ninformative null comparison.\nNo full anatomical characterization.',va='center')
        else:
            one_slice(ax,reconstruct(patterns[(cohort,family,'common')]['haufe'],mask),mask,ref,
                      'Haufe pattern\n'+('FRIEND − STRANGER' if family.startswith('closeness') else 'POSITIVE − NEGATIVE'))
    fig.suptitle('Figure 4. Closeness and valence — primary development N='+str(len(scopes[(PRIMARY,cohort)]['subjects'])),fontsize=14)
    fig.tight_layout(); save(out,'Figure4_closeness_valence',fig)


def expression_figures(out,scopes):
    tables=[pd.read_csv(out.output(f'work/{cohort}_oof_scores.tsv'),sep='\t') for cohort in TARGETS]
    scores=pd.concat(tables,ignore_index=True)
    groups=[(keys,g.sort_values('margin')) for keys,g in scores.groupby(['cohort','family','task'],sort=False)
            if keys[1]!='closeness_reward']
    fig,axes=plt.subplots(4,3,figsize=(13,15)); margins,m_axes=plt.subplots(4,3,figsize=(13,15)); summary=[]
    for ax,m_ax,(keys,g) in zip(axes.flat,m_axes.flat,groups):
        cohort,family,task=keys; positive,negative=DIRECTION[family]
        for r in g.itertuples(): ax.plot([0,1],[r.negative_score,r.positive_score],color='#335c81',alpha=.10,lw=.5)
        ax.violinplot([g.negative_score,g.positive_score],positions=[0,1],showmedians=True,showextrema=False)
        ax.set_xticks([0,1],[negative,positive],fontsize=8); ax.set_ylabel('OOF decision score')
        title=FAMILY[family]+'\n'+LABELS[task]+f'; N={len(g)}'; ax.set_title(title,fontsize=9)
        values=g.margin.to_numpy(float); rng=np.random.default_rng(20260928)
        means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(10000)])
        lo,hi=np.quantile(means,[.025,.975]); y=np.linspace(-.15,.15,len(values))
        m_ax.scatter(values,y,s=9,alpha=.45,color='#335c81'); m_ax.axvline(0,color='gray',ls='--')
        m_ax.errorbar(values.mean(),.25,xerr=[[values.mean()-lo],[hi-values.mean()]],fmt='o',color='#bb5522',capsize=4,label='Mean ± bootstrap 95% CI')
        m_ax.scatter(np.median(values),-.25,marker='D',color='black',label='Median',s=22)
        m_ax.set(title=title,xlabel='OOF positive − negative margin',ylim=(-.4,.45),yticks=[])
        if len(summary)==0: m_ax.legend(fontsize=7,loc='upper left')
        summary.append({'cohort':cohort,'family':family,'task':task,'n':len(g),'positive_fraction':float((values>0).mean()),
                        'mean':values.mean(),'median':np.median(values),'mean_ci_low':lo,'mean_ci_high':hi,
                        'q05':np.quantile(values,.05),'q95':np.quantile(values,.95),'minimum':values.min(),'maximum':values.max()})
    fig.suptitle('Figure 5. Paired out-of-fold expression; each faint line is one development participant',fontsize=13)
    margins.suptitle('Figure 5 supplement. Individual margins and participant-bootstrap mean intervals',fontsize=13)
    fig.tight_layout(); margins.tight_layout(); save(out,'Figure5_oof_expression',fig); save(out,'Figure5_oof_margins',margins)
    write_tsv(out,'results/aggregate/oof_margin_summary.tsv',summary)


def reliability_figures(out,scopes):
    families=['social_context','social_reward','closeness_positive','closeness_negative','closeness_reward','valence']
    fig,axes=plt.subplots(6,2,figsize=(11,22)); summary=[]
    for i,family in enumerate(families):
        cohort='three_paradigm' if family in TARGETS['three_paradigm'] else 'partner_pair'
        runs=scopes[(PRIMARY,cohort)]['runs']
        for j,task in enumerate(('sharedreward','trust')):
            g=runs[(runs.family==family)&(runs.test_task==task)].pivot(index='subject',columns='run',values='margin').reindex(columns=[1,2]).dropna()
            x,y=g[1].to_numpy(float),g[2].to_numpy(float); valid=len(g)>1 and np.std(x)>0 and np.std(y)>0
            r=float(pearsonr(x,y).statistic) if valid else np.nan; rho=float(spearmanr(x,y).statistic) if valid else np.nan
            icc=icc31(g.to_numpy(float)); ax=axes[i,j]; ax.scatter(x,y,s=12,alpha=.5,color='#335c81')
            limits=[min(x.min(),y.min()),max(x.max(),y.max())]; ax.plot(limits,limits,'--',color='gray',lw=.8)
            if valid:
                slope,intercept=np.polyfit(x,y,1); ax.plot(limits,np.array(limits)*slope+intercept,color='#bb5522')
            ax.set(xlabel='Run 1 OOF margin',ylabel='Run 2 OOF margin',title=FAMILY[family]+' — '+LABELS[task]+f'\nN={len(g)}; r={r:.2f}; ρ={rho:.2f}; ICC(3,1)={icc:.2f}')
            summary.append({'cohort':cohort,'family':family,'task':task,'n':len(g),'pearson_r':r,'spearman_rho':rho,'icc_3_1':icc})
    fig.suptitle('Figure 6. Run-level reliability of individual OOF margins\nStrong condition decoding does not establish stable individual differences',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95)); save(out,'Figure6_run_reliability',fig); write_tsv(out,'results/aggregate/reliability.tsv',summary)


def qc_figure(out,robustness):
    fig,ax=plt.subplots(figsize=(7,6))
    for family,g in robustness.groupby('family',sort=False):
        ax.scatter(g.primary_accuracy,g.sensitivity_accuracy,label=FAMILY[family],s=45)
        for r in g.itertuples(): ax.annotate(LABELS[r.task],(r.primary_accuracy,r.sensitivity_accuracy),fontsize=7,xytext=(3,3),textcoords='offset points')
    ax.plot([0,1],[0,1],'--',color='gray'); ax.axvline(.5,color='gray',lw=.5); ax.axhline(.5,color='gray',lw=.5)
    ax.set(xlim=(.35,1),ylim=(.35,1),xlabel='Primary QC OOF accuracy',ylabel='Four-metric QC OOF accuracy',title='QC robustness: existing common models\nDifferent samples/masks; descriptive, not replication')
    ax.legend(fontsize=8,loc='lower right'); fig.tight_layout(); save(out,'supplement_qc_robustness',fig)


def permutation_figures(out,summary,nulls):
    for key,(tasks,null) in nulls.items():
        cohort,family,mode=key; fig,axes=plt.subplots(1,len(tasks),figsize=(4*len(tasks),3.5),squeeze=False)
        for j,task in enumerate(tasks):
            r=summary[(summary.cohort==cohort)&(summary.family==family)&(summary.scope==mode)&(summary.test_task==task)].iloc[0]
            ax=axes[0,j]; ax.hist(null[:,j],bins=np.linspace(0,1,31),color='#6687a0',alpha=.85)
            ax.axvline(.5,color='gray',ls='--',label='Chance'); ax.axvline(r.observed_accuracy,color='#b34129',lw=2,label='Observed OOF')
            ax.set(xlim=(0,1),xlabel='Permuted-label OOF accuracy',ylabel='Permutations',title=LABELS[task]+f'\np={r.empirical_p:.4f}; B={int(r.permutations)}')
            ax.legend(fontsize=7)
        fig.suptitle(FAMILY[family]+' / '+mode+' — participant-label sanity check',fontsize=12)
        fig.tight_layout(); save(out,'permutation_'+cohort+'_'+family+'_'+mode,fig)
