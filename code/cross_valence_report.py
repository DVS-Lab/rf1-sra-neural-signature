"""Figures and cautious reports for the prespecified development follow-up."""
import json
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cross_valence_design import domains,FAMILIES
from cross_valence_anatomy import standard_image
from characterization_audit import require,PRIMARY
from characterization_compute import assert_development
from reporting import md_table
from utils import atomic_output,write_text,write_json,sha256

LABELS={'sharedreward_positive':'SR +','sharedreward_negative':'SR −','trust_positive':'Trust +','trust_negative':'Trust −',
        'socialdoors_positive':'Doors +','socialdoors_negative':'Doors −'}
TITLE={'social_context':'Social context','friend_stranger_context':'Friend–stranger context'}
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','font.size':10})


def save(out,fig,name):
    for suffix in ('png','pdf','svg'):
        with atomic_output(out,'results/figures/'+name+'.'+suffix) as p: fig.savefig(p,dpi=300,bbox_inches='tight')
    plt.close(fig)


def factorial(out):
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    for ax,(first,second,name) in zip(axes,[('Social','Nonsocial','Sociality'),('Friend','Stranger','Relationship context')]):
        ax.axis('off'); ax.set_title(name,pad=20,fontsize=16)
        table=ax.table(cellText=[[first+' positive (A)',first+' negative (B)'],[second+' positive (C)',second+' negative (D)']],
            rowLabels=[first,second],colLabels=['Positive outcome','Negative outcome'],cellLoc='center',bbox=[.16,.56,.82,.30])
        table.auto_set_font_size(False); table.set_fontsize(10)
        for (row,col),cell in table.get_celld().items():
            cell.set_edgecolor('#6f8791'); cell.set_facecolor('#eaf2f5' if row==0 or col==-1 else 'white')
        text=('Context main effect: (A + B)/2 − (C + D)/2\n\n'
              'Positive context: A − C     Negative context: B − D\n\n'
              'Valence main effect: (A + C)/2 − (B + D)/2\n\n'
              'Context × valence interaction: (A − B) − (C − D)')
        ax.text(.02,.43,text,va='top',fontsize=11,linespacing=1.25)
    fig.suptitle('Figure A. Main effects and interactions answer different questions',fontsize=18)
    fig.text(.5,.01,'Conceptual equal-category factorial contrast. The reused v4 valence comparator averages computer/friend/stranger equally.',ha='center',fontsize=10)
    save(out,fig,'FigureA_factorial')


def matrix(out,performance,cohort,family,name):
    frame=performance[(performance.cohort==cohort)&(performance.family==family)&(performance.scope=='matrix')]
    names=domains(cohort); values=frame.pivot(index='train_domain',columns='test_domain',values='accuracy').reindex(index=names,columns=names)
    fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(values,vmin=0,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(len(names)),[LABELS[s] for s in names],rotation=25,ha='right')
    ax.set_yticks(range(len(names)),[LABELS[s] for s in names])
    for i,a in enumerate(names):
        for j,b in enumerate(names):
            task=a.rsplit('_',1)[0]!=b.rsplit('_',1)[0]; val=a.rsplit('_',1)[1]!=b.rsplit('_',1)[1]
            code='TV' if task and val else 'T' if task else 'V' if val else 'W'
            value=values.iloc[i,j]; ax.text(j,i,f'{value:.1%}\n{code}',ha='center',va='center',color='white' if abs(value-.5)>.3 else 'black')
    ax.set(xlabel='Test domain: unseen participants',ylabel='Training domain',title=f'{TITLE[family]}; N={int(frame.n.iloc[0])}; chance=50%')
    fig.colorbar(im,ax=ax,label='OOF paired forced-choice accuracy')
    fig.text(.5,.005,'W: same domain   T: task changes   V: valence changes   TV: both change',ha='center')
    fig.tight_layout(rect=[0,.03,1,1]); save(out,fig,name)


def pooled(out,perf):
    fig,axes=plt.subplots(1,2,figsize=(14,7),sharex=True)
    for ax,family in zip(axes,FAMILIES):
        frame=perf[(perf.cohort=='partner_pair')&(perf.family==family)&perf.train_domain.isin(['pooled_positive','pooled_negative','collapsed_common'])]
        labels=[]; vals=[]
        for train in ('pooled_positive','pooled_negative','collapsed_common'):
            group=frame[frame.train_domain==train]
            for row in group.itertuples():
                prefix={'pooled_positive':'Positive → negative','pooled_negative':'Negative → positive','collapsed_common':'Collapsed context'}[train]
                target=row.test_domain.replace('sharedreward','SR').replace('trust','Trust').replace('_positive','').replace('_negative','')
                labels.append(prefix+' | '+target); vals.append((row.accuracy,row.ci_low,row.ci_high))
        v=np.array(vals); y=np.arange(len(v))
        ax.errorbar(v[:,0],y,xerr=np.stack([v[:,0]-v[:,1],v[:,2]-v[:,0]]),fmt='o',capsize=3,color='#276779')
        ax.axvline(.5,color='gray',ls='--'); ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(0,1)
        ax.set_title(TITLE[family]); ax.set_xlabel('OOF accuracy and descriptive 95% CI')
    fig.suptitle('Figure D. Pooled cross-valence and collapsed-context models; N=192',fontsize=16)
    fig.tight_layout(rect=[0,.04,1,.94]); fig.text(.5,.015,'Combined endpoints average task correctness within each participant; intervals resample participants.',ha='center')
    save(out,fig,'FigureD_pooled')


def image_panel(ax,data,bg,ref,axis,coordinate,title):
    # Both arrays have already been canonicalized to the same standard-space grid.
    point=np.zeros(3); point[axis]=coordinate
    index=int(np.rint(nib.affines.apply_affine(np.linalg.inv(ref.affine),point)[axis]))
    index=int(np.clip(index,0,data.shape[axis]-1))
    plane=np.take(data,index,axis=axis).T; under=np.take(bg,index,axis=axis).T
    positive=bg[bg>0]; upper=float(np.percentile(positive,99)) if len(positive) else 1
    other=[i for i in range(3) if i!=axis]; zooms=ref.header.get_zooms()[:3]
    aspect=zooms[other[1]]/zooms[other[0]]
    ax.imshow(under,origin='lower',cmap='gray',vmin=0,vmax=upper,aspect=aspect)
    bound=float(np.max(np.abs(data))) or 1
    image=ax.imshow(np.ma.masked_where(plane==0,plane),origin='lower',cmap='RdBu_r',vmin=-bound,vmax=bound,alpha=.8,aspect=aspect)
    ax.set_title(title,fontsize=10); ax.axis('off'); return image


def brains(out,scope,assets,spatial):
    assert_development(scope)
    bg_img=standard_image(scope,assets['background'],assets['background_sha256'])
    ref_path=out.output('results/maps/partner_pair_analysis_mask.nii.gz')
    reference=nib.load(ref_path)
    bg_grid=resample_from_to(bg_img,(reference.shape,reference.affine),order=1)
    bg=nib.as_closest_canonical(bg_grid).get_fdata()
    volumes={}
    for family in FAMILIES:
        for mode in ('positive','negative','collapsed'):
            for kind in ('weights','haufe','mean_difference'):
                assert_development(scope)
                p=out.output(f'results/maps/partner_pair_{family}_{mode}_common_DEV_{kind}.nii.gz')
                image=nib.as_closest_canonical(nib.load(p)); volumes[(family,mode,kind)]=(image.get_fdata(),image)
    fig,axes=plt.subplots(2,4,figsize=(16,8),gridspec_kw={'width_ratios':[1,1,1,1.1]})
    for i,family in enumerate(FAMILIES):
        for j,mode in enumerate(('positive','negative','collapsed')):
            data,ref=volumes[(family,mode,'haufe')]
            image=image_panel(axes[i,j],data,bg,ref,0,0,TITLE[family]+'\n'+mode+'; x≈0 mm')
            fig.colorbar(image,ax=axes[i,j],shrink=.5)
        mat=np.eye(3); labels=['positive','negative','collapsed']
        for row in spatial[(spatial.family==family)&(spatial.kind=='haufe')].itertuples():
            a,b=labels.index(row.pattern_a),labels.index(row.pattern_b); mat[a,b]=mat[b,a]=row.spatial_r
        ax=axes[i,3]; im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(3),labels,rotation=25,ha='right'); ax.set_yticks(range(3),labels); ax.set_title('Haufe spatial correlation')
        for a in range(3):
            for b in range(3): ax.text(b,a,f'{mat[a,b]:.2f}',ha='center',va='center',color='white' if abs(mat[a,b])>.7 else 'black')
    fig.suptitle('Figure E. Descriptive forward patterns; red → social / friend, blue → nonsocial / stranger',fontsize=15)
    fig.text(.5,.02,'Unthresholded; standard anatomical background; separate color ranges; no regional significance or necessity implied.',ha='center')
    fig.tight_layout(rect=[0,.05,1,.93]); save(out,fig,'FigureE_patterns')
    for family in FAMILIES:
        for kind in ('weights','haufe','mean_difference'):
            fig,axes=plt.subplots(3,3,figsize=(12,10))
            for i,mode in enumerate(('positive','negative','collapsed')):
                data,ref=volumes[(family,mode,kind)]
                for j,(axis,coord,label) in enumerate([(0,0,'sagittal x≈0'),(1,-50,'coronal y≈−50'),(2,20,'axial z≈20')]):
                    image=image_panel(axes[i,j],data,bg,ref,axis,coord,mode+' | '+label+' mm')
                    fig.colorbar(image,ax=axes[i,j],shrink=.5)
            fig.suptitle(TITLE[family]+' — '+kind+' (descriptive)\nPositive → social / friend; negative → nonsocial / stranger; axial/coronal L→R',fontsize=14)
            fig.tight_layout(rect=[0,0,1,.93]); save(out,fig,'supplement_'+family+'_'+kind)
    fig,axes=plt.subplots(2,3,figsize=(12,8)); labels=['positive','negative','collapsed']
    for i,family in enumerate(FAMILIES):
        for j,kind in enumerate(('weight','haufe','mean_difference')):
            mat=np.eye(3)
            for row in spatial[(spatial.family==family)&(spatial.kind==kind)].itertuples():
                a,b=labels.index(row.pattern_a),labels.index(row.pattern_b); mat[a,b]=mat[b,a]=row.spatial_r
            ax=axes[i,j]; ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r'); ax.set_title(TITLE[family]+'\n'+kind)
            ax.set_xticks(range(3),labels,rotation=20,ha='right'); ax.set_yticks(range(3),labels)
            for a in range(3):
                for b in range(3): ax.text(b,a,f'{mat[a,b]:.2f}',ha='center',va='center',color='white' if abs(mat[a,b])>.7 else 'black')
    fig.tight_layout(); save(out,fig,'supplement_spatial_similarity')


def permutation_figures(out,perm):
    for (family,train),group in perm.groupby(['family','train_domain'],sort=False):
        state=json.loads(out.output('work/permutations/'+family+'_'+train+'.json').read_text()); null=np.array(state['null'])
        fig,axes=plt.subplots(1,len(group),figsize=(5*len(group),3.8),squeeze=False)
        for j,(ax,row) in enumerate(zip(axes[0],group.itertuples())):
            ax.hist(null[:,j],bins=np.linspace(0,1,26),color='#81a2b3')
            ax.axvline(.5,color='gray',ls='--',label='Chance'); ax.axvline(row.observed_accuracy,color='#bb472b',label='Observed OOF')
            ax.set(xlim=(0,1),xlabel='Permuted-label OOF accuracy',ylabel='Permutations',
                   title=f'{row.test_domain}\np={row.empirical_p:.4f}; B={row.permutations}')
            ax.legend(fontsize=8)
        fig.suptitle(TITLE[family]+' | '+train); fig.tight_layout(rect=[0,0,1,.9]); save(out,fig,'permutation_'+family+'_'+train)


def freeze(out,spec):
    metadata=json.loads(out.output('provenance/models.json').read_text())
    entries=[x for x in metadata['models'] if x['mode']=='collapsed' and x['model']=='common']
    body='# Candidate signatures for prospective validation review\n\n'
    body+='These exact development artifacts are candidates, not validated signatures or automatic winners. Review the directional performance, uncertainty, permutation diagnostics, construct boundaries and intended endpoints before preregistration. No single accuracy cutoff selects a model.\n\n'
    for e in entries:
        payload={**e,'classifier':spec['classifier'],'random_state':spec['seed'],
                 'normalization':'per-map spatial mean centering inside the original partner-pair mask; no feature scaling',
                 'cohort':'tsnr_coverage_fd/partner_pair','fingerprint':metadata['fingerprint'],'holdout_scored':False}
        relative='provenance/candidate_'+e['family']+'.json'; write_json(out,relative,payload)
        body+='## '+TITLE[e['family']]+'\n\n'
        if e['family']=='social_context':
            body+='Reused original v4 common model. SR: mean[(F_rew+F_pun)/2, (S_rew+S_pun)/2] versus (C_rew+C_pun)/2; Trust: replace rew/pun with rec/def. Positive class = human context; negative class = computer context. SR COPEs: C=2/1, F=4/3, S=6/5; Trust: C=5/4, F=7/6, S=9/8 (positive/negative).\n\n'
        else:
            body+='New pooled valence-collapsed candidate. SR: (F_rew+F_pun)/2 versus (S_rew+S_pun)/2; Trust: (F_rec+F_def)/2 versus (S_rec+S_def)/2. Positive class = friend; negative class = stranger. SR COPEs: F=4/3, S=6/5; Trust: F=7/6, S=9/8 (positive/negative). Cross-valence support must be assessed from the full directional results, not assumed from the candidate file existing.\n\n'
        body+=f'Training: all {e["training_n"]} primary development participants; both tasks; no protected validation participant. Shared Reward full-trial retained-run strategy and Trust outcome inputs are unchanged.\n\n'
        for label,key in [('Weight file','weight_path'),('Weight SHA256','weight_sha256'),('Mask','mask_path'),('Mask SHA256','mask_sha256'),('Intercept','intercept')]:
            body+=f'- {label}: `{e[key]}`\n'
        body+=f'- Exact model manifest: `{out.output(relative).relative_to(out.root)}`\n- Model manifest SHA256: `{sha256(out.output(relative))}`\n'
        body+='\nPreprocessing: independent spatial mean centering inside the frozen partner-pair mask; no PCA, feature selection or scaling. LinearSVC(C=1, dual=True, class_weight=None, max_iter=100000, tol=.0001, random_state=20260928). NIfTI coefficients are float32; the manifest preserves the fitted intercept.\n\n'
    body+='## Comparators and boundaries\n\nOutcome valence, sociality × valence interaction, friend–stranger × valence interaction, and Doors transfer remain explicit comparisons/boundaries. Their completed v4 results are not retuned.\n\n'
    body+='Before scoring N=50: specify exact models, contrasts, transfer endpoints, success criteria and multiplicity; review whether a separately approved visual-control analysis is needed. Neither pure closeness, abstract social representation, trait-like social sensitivity, nor a social reward signature is established automatically.\n\nholdout_scored=False.\n'
    write_text(out,'reports/FREEZE_CANDIDATES.md',body)


def report(out,perf,spatial,perm,assets):
    primary=perf[perf.cohort=='partner_pair']; cols=['train_domain','test_domain','n','accuracy','ci_low','ci_high','mean_margin','margin_ci_low','margin_ci_high']
    body='# Final development follow-up: cross-valence context\n\n'
    body+='Development only. Participants, five folds, QC, masks, source images, centering and LinearSVC settings remain fixed. Partner-pair N='+str(int(primary.n.iloc[0]))+'. Three-paradigm N='+str(int(perf[perf.cohort=='three_paradigm'].n.iloc[0]))+'. Primary validation N=50 remains untouched. See [audit](IMPLEMENTATION_AUDIT.md).\n\n'
    for family in FAMILIES:
        f=primary[(primary.family==family)&(primary.scope=='matrix')]
        same=[]; both=[]
        for row in f.itertuples():
            a,av=row.train_domain.rsplit('_',1); b,bv=row.test_domain.rsplit('_',1)
            if av!=bv: (same if a==b else both).append(row.Index)
        body+='## '+TITLE[family]+': does it generalize across valence and across both task and valence?\n\n'
        for label,indices in [('Same task, cross valence',same),('Both task and valence change',both)]:
            group=f.loc[indices]; above=int((group.ci_low>.5).sum())
            body+=f'{label}: observed accuracy ranges from {group.accuracy.min():.1%} to {group.accuracy.max():.1%}; {above}/{len(group)} descriptive 95% intervals lie entirely above chance. These intervals summarize OOF participants, not independent training-fold replications. Assess all directions; asymmetry limits invariance claims.\n\n'+md_table(group[cols])+'\n\n'
        pool=primary[(primary.family==family)&(primary.scope=='pooled_cross_valence')]
        body+='Training pooled across SR and Trust at one valence, testing exclusively at the other:\n\n'+md_table(pool[cols])+'\n\n'
    collapsed=primary[(primary.family=='friend_stranger_context')&(primary.scope=='collapsed')]
    body+='## Does explicit valence-collapsed friend–stranger context classify both tasks?\n\n'+md_table(collapsed[cols])+'\n\n'
    body+='This model averages outcomes before centering, then distinguishes friend from stranger. It may encode familiarity, identity, relationship status, visual or relationship-linked semantic information; it is not labeled pure closeness.\n\n'
    body+='## Are positive, negative and collapsed spatial patterns similar?\n\n'+md_table(spatial)+'\n\n'
    body+='These spatial correlations describe patterns from overlapping development data and do not themselves demonstrate prediction or independent replication. Haufe patterns are Cov(X,score)/Var(score), without feature z-scoring. Raw weights are discriminative coefficients, not regional activation or necessity. All anatomical displays are descriptive and use fixed x=0, y=-50, z=20 mm cuts in the supplements.\n\n'
    body+=f'Standard display background: `{assets["background"]}`; SHA256 `{assets["background_sha256"]}`. FSL MNI backgrounds/atlases are descriptive display/label references and need not match the exact nonlinear template variant. No atlas or background was downloaded.\n\n'
    body+='## Main effects versus the existing interactions\n\n'
    compare=pd.read_csv(out.output('results/aggregate/frozen_v4_comparators.tsv'),sep='\t')
    body+=md_table(compare[compare.scope=='common'][['family','test_task','n','accuracy','ci_low','ci_high']])+'\n\n'
    body+='Context averages outcomes; positive-context discrimination compares social wins to nonsocial wins; the sociality × valence interaction compares the positive-minus-negative difference between categories. Context-to-interaction prediction is not a success criterion. Weak interaction decoding does not show that social/friend outcomes lack reward value; it indicates limited decodable differential outcome modulation under these tasks/models. Valence is a reused v4 comparator averaging C/F/S equally, whereas Figure A illustrates an equal-category factorial contrast.\n\n'
    body+='## Does Doors share the organization or remain a boundary?\n\n'
    doors=perf[(perf.cohort=='three_paradigm')&(perf.scope=='matrix')]
    body+=md_table(doors[doors.test_domain.str.startswith('socialdoors')][cols])+'\n\n'
    body+='Read the two within-Doors cross-valence cells separately from SR/Trust-to-Doors transfer. This secondary N=178 matrix does not redefine the primary N=192 cohort, construct or feature space. The full six-domain matrix is supplied in the supplement.\n\n'
    body+='## Prespecified participant-label permutation diagnostics\n\n'+md_table(perm)+'\n\n'
    body+='One participant flip is shared across all tasks/valences and applied to training and test labels; five folds are refit. The 12 prespecified endpoints use 500 permutations each (shared fits where endpoints have identical training data). Empirical p=(1+null≥observed)/(1+B), with resolution 1/501. These are development diagnostics without project-wide multiplicity control. Combined-task accuracy averages correctness within each participant; it is not the fraction with a positive task-averaged margin.\n\n'
    body+=('All null means are within the descriptive [.45,.55] check.\n\n' if not perm.chance_centering_flag.any() else '**REVIEW REQUIRED: at least one null mean is outside [.45,.55]; do not claim all sanity checks passed.**\n\n')
    body+='## Alternate QC and remaining interpretation limits\n\nNo new cross-valence models were run under the alternate QC policy. Earlier characterization robustness cannot establish robustness of these new endpoints. Shared Reward is full-trial; Trust and Doors are outcome-phase. Perceptual/identity/familiarity and phase differences remain alternatives. Existing weak run-level reliability still limits individual-difference or trait claims.\n\n'
    body+='## Which exact signatures could be frozen?\n\nSee [FREEZE_CANDIDATES.md](FREEZE_CANDIDATES.md) for the original pooled SR–Trust social-context model and the new pooled valence-collapsed friend–stranger model, exact inputs, coefficients, intercepts, masks and SHA256 fingerprints. Review the results before deciding endpoints; no automatic accuracy cutoff selects a winner. Neither candidate has been validated.\n\n'
    body+='See [VISUAL_SENSITIVITY_PROPOSAL.md](VISUAL_SENSITIVITY_PROPOSAL.md) for a local independent atlas inventory and proposed exclusion. **No visual exclusion analysis was executed.**\n\n'
    for name in ['FigureA_factorial','FigureB_social_context','FigureC_friend_stranger','FigureD_pooled','FigureE_patterns','supplement_Doors_matrix','supplement_spatial_similarity']:
        body+=f'- [{name} PNG](../../../results/revised/cross_valence/figures/{name}.png) · [PDF](../../../results/revised/cross_valence/figures/{name}.pdf)\n'
    body+='\nPNG figures use 300 dpi; charts retain vector text in PDF/SVG. Original v4 and characterization artifacts are checked unchanged after rendering. **holdout_scored=False; primary validation N=50 untouched.**\n'
    write_text(out,'reports/REPORT.md',body)


def render(out,scopes,spec,assets):
    scope=scopes[(PRIMARY,'partner_pair')]; assert_development(scope)
    perf=pd.read_csv(out.output('results/aggregate/performance.tsv'),sep='\t')
    spatial=pd.read_csv(out.output('results/aggregate/spatial_similarity.tsv'),sep='\t')
    perm=pd.read_csv(out.output('results/aggregate/permutation_summary.tsv'),sep='\t')
    factorial(out)
    matrix(out,perf,'partner_pair','social_context','FigureB_social_context')
    matrix(out,perf,'partner_pair','friend_stranger_context','FigureC_friend_stranger')
    matrix(out,perf,'three_paradigm','social_context','supplement_Doors_matrix')
    pooled(out,perf); brains(out,scope,assets,spatial); permutation_figures(out,perm)
    compare=pd.read_csv(out.output('results/aggregate/frozen_v4_comparators.tsv'),sep='\t').query('scope == "common"')
    fig,ax=plt.subplots(figsize=(9,5))
    labels=[r.family+' | '+r.test_task for r in compare.itertuples()]; a=compare.accuracy.to_numpy()
    ax.errorbar(a,np.arange(len(a)),xerr=np.stack([a-compare.ci_low.to_numpy(),compare.ci_high.to_numpy()-a]),fmt='o',capsize=3)
    ax.set_yticks(range(len(a)),labels); ax.invert_yaxis(); ax.axvline(.5,ls='--',color='gray'); ax.set_xlim(0,1)
    ax.set_title('Frozen v4 common-model comparators; N=192'); ax.set_xlabel('OOF accuracy and descriptive 95% CI'); fig.tight_layout()
    save(out,fig,'supplement_interaction_comparators')
    freeze(out,spec); report(out,perf,spatial,perm,assets)
