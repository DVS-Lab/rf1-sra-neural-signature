"""Evidence-linked characterization report; no causal or inferential anatomy claims."""
import pandas as pd
from characterization_audit import PRIMARY, TARGETS
from reporting import md_table
from utils import write_text


def report(out,scopes,spec):
    def read(name): return pd.read_csv(out.output('results/aggregate/'+name+'.tsv'),sep='\t')
    corr=read('weight_pattern_correlations'); task=read('task_pattern_similarity'); cross=read('cross_family_similarity')
    perm=read('permutation_summary'); qc=read('qc_robustness'); reliability=read('reliability'); clusters=read('descriptive_clusters')
    margins=read('oof_margin_summary')
    common=corr[corr.model=='common']
    to_common=task[(task.model_b=='common')&(task.model_a!='common')]
    pair=task[(task.model_a!='common')&(task.model_b!='common')&(task.model_a<task.model_b)]
    ready=perm[(perm.family!='social_reward')&(~perm.chance_centering_flag)&(perm.empirical_p<=.05)]
    issue=perm[perm.chance_centering_flag]
    body='# Frozen v4 development-result characterization\n\n'
    body+=('This report characterizes the existing primary `tsnr_coverage_fd` development results. '
           'No original assignments, folds, masks, contrasts, QC thresholds, or classifier settings were changed. '
           'The four-metric policy is a sensitivity analysis. No validation data were loaded or scored.\n\n')
    body+='## Audit and sample\n\n'
    body+='See [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md). Private membership and frozen hashes passed; selected saved fold maps reproduced OOF margins and classifications before scientific characterization. '
    body+=f"Primary development N={len(scopes[(PRIMARY,'partner_pair')]['subjects'])} for the partner pair and N={len(scopes[(PRIMARY,'three_paradigm')]['subjects'])} for three paradigms. QC-qualified primary validation N=50 remains untouched.\n\n"
    body+='## What the decoders track spatially\n\n'
    body+=('Raw LinearSVC coefficients are discriminative weights, not activation or independent regional effects. '
           'Haufe patterns describe covariance between the exact centered training maps and the fitted decoder output, divided by output variance. '
           'They are descriptive forward patterns, not inferential or causal maps. Mean-difference maps describe group-average class differences. '
           'The correlations below show whether these three descriptions align; they do not establish a psychological mechanism.\n\n')
    body+=md_table(common)+'\n\n'
    body+='## Common versus task-specific patterns\n\n'
    body+='Task-specific patterns are compared on their fixed common mask. Correlations with the pooled pattern indicate resemblance, not proof that one task dominates the decoder; correlated predictors and task scaling can affect raw weights.\n\n'
    body+=md_table(pair)+'\n\nCommon-model comparisons:\n\n'+md_table(to_common)+'\n\n'
    for family in ('social_context','social_reward'):
        g=to_common[(to_common.family==family)&(to_common.kind=='haufe')].dropna(subset=['spatial_r'])
        if len(g):
            best=g.loc[g.spatial_r.idxmax()]; worst=g.loc[g.spatial_r.idxmin()]
            body+=f"For {family}, the common Haufe pattern is most similar to {best.model_a} (r={best.spatial_r:.3f}) and least similar to {worst.model_a} (r={worst.spatial_r:.3f}). This is a descriptive spatial comparison, not a task-contribution estimate.\n\n"
    body+='## Context, closeness, reward modulation, and valence\n\n'
    body+=('The performance dissociation must retain its original meaning: social context averages across outcome valence; social reward compares human versus nonsocial positive-minus-negative modulation; closeness is friend versus stranger; valence is positive versus negative outcome. '
           'Strong context and friend–stranger decoding cannot be relabeled as a task-general social-reward signature. Failed reward decoding is not proof of no reward-related information.\n\n')
    body+='Common-model cross-family similarities use intersecting valid masks (resampling only for comparison if needed):\n\n'+md_table(cross)+'\n\n'
    close=cross[(cross.family_a=='closeness_positive')&(cross.family_b=='closeness_negative')]
    for row in close.itertuples(): body+=f'Positive- versus negative-outcome closeness {row.kind}: r={row.spatial_r:.3f}.\n\n'
    body+='## Descriptive spatial locations and alternatives\n\n'
    body+=(f'Bootstrap sign stability uses {spec["bootstrap_samples"]} participant resamples and P(positive) or P(negative) ≥{spec["sign_stability"]}. '
           f'Components have at least {spec["minimum_cluster_voxels"]} face-connected voxels. Peaks and means below refer to the bootstrap-mean Haufe pattern. '
           'No FWE/FDR correction or cluster p value is implied. Display thresholds (top 5%/1%) are also descriptive. No atlas was downloaded; coordinates alone do not establish a network assignment.\n\n')
    body+=md_table(clusters.sort_values('peak_absolute_haufe',ascending=False).head(40))+'\n\n' if len(clusters) else 'No components met the descriptive size/sign-stability rule. This is not a statistical null result.\n\n'
    body+=('Inspect Figure 3 and the full slice mosaics alongside these coordinates for occipital/ventral versus broader distributions. '
           'An automated coordinate table cannot resolve sensory versus abstract social coding, so anatomical/network attribution remains a visual-review item. '
           'Friend–stranger patterns can reflect closeness, familiarity, person identity, perceptual information, and relationship-linked representations. '
           'Shared Reward is full-trial; Trust and Doors are outcome-phase. Shared visual content and different event timing remain alternative explanations. '
           'If visual review indicates a predominantly perceptual pattern, an independently defined, prospectively justified visual-control sensitivity analysis is a next-step option; none was implemented here.\n\n')
    body+='## Participant-label permutation sanity checks\n\n'
    body+=(f'{spec["permutations"]} permutations refit the relevant five-fold classifiers with one random label flip per participant, shared across that participant’s tasks. Both training and test labels are permuted. '
           'Folds, features, masks and parameters remain fixed. Empirical upper-tail p=(1+null≥observed)/(1+B); these diagnostic values do not replace the original inferential framework or provide project-wide multiplicity control.\n\n')
    body+=md_table(perm)+'\n\n'
    if len(issue): body+='**Sanity review required:** at least one null mean differs from .50 by more than .05. This descriptive centering flag must be investigated before claiming that the sanity check passed.\n\n'
    else: body+='All empirical null means are within .05 of .50 under the prespecified descriptive centering check. This checks the label-scrambling implementation, not every possible confound.\n\n'
    body+='## OOF score distributions\n\n'
    body+=('Figure 5 uses only verified out-of-fold scores. Final in-sample DEV scores are used only in the descriptive Haufe transformation, never as accuracy evidence. '
           'Forced-choice accuracy counts the sign of each participant’s margin equally: extreme margins cannot by magnitude alone create high accuracy. Means can be influenced by extremes; medians, individual margins, and quantiles are provided below.\n\n')
    body+=md_table(margins)+'\n\n'
    body+='## QC robustness\n\n'+md_table(qc)+'\n\n'
    body+='These compare already-completed models; there was no refitting for the QC comparison. Differences can reflect both sample and mask changes. Similarity across QC policies is descriptive robustness, not independent replication.\n\n'
    body+='## Run-level individual-difference reliability\n\n'+md_table(reliability)+'\n\n'
    body+=('High condition-level forced-choice decoding and stable ranking of individuals are distinct properties. Low run-to-run margin correlations/ICC limit interpretation of a score as an individual-difference measure even when class ordering is accurate. '
           'These margins also come from fold-specific decoders; their score scaling and reliability should be considered before using a single score for between-person inference.\n\n')
    body+='## Readiness for validation\n\n'
    body+='The following observed development cells exceed their diagnostic permutation distributions at nominal p≤.05 with centered nulls; this is a candidate list for preregistration, not validation success:\n\n'
    body+=md_table(ready[['family','scope','test_task','observed_accuracy','empirical_p']])+'\n\n'
    body+=('Prioritize condition discrimination claims supported by task-transfer, spatial review and sanity checks. Broad three-paradigm generalization requires inspecting the held-out Doors result specifically. '
           'Individual-difference claims require stronger reliability evidence. Select final constructs, transfer directions, endpoints and multiplicity before any later validation decision. '
           'No internal/external validation scoring or anatomical exclusion-mask analysis is implemented here.\n\n')
    body+='## Figure package\n\n'
    names=['Figure1_schematic','Figure2_context_reward_transfer','Figure3_social_context_pattern','Figure4_closeness_valence','Figure5_oof_expression','Figure6_run_reliability']
    for i,name in enumerate(names,1): body+=f'- [Figure {i} PNG](../../../results/revised/characterization/figures/{name}.png) · [PDF](../../../results/revised/characterization/figures/{name}.pdf)\n'
    body+='\nSupplemental figures include OOF margins, QC robustness, all permutation nulls, raw/Haufe/mean maps and task/common spatial similarities. Charts retain vector text in PDF/SVG; PNG files use 300 dpi.\n\n'
    body+='Original v4 hashes are checked again after plotting. Check `provenance/revised/characterization/run_status.json` for final completion.\n\n**holdout_scored = False. Validation N=50 remains untouched.**\n'
    write_text(out,'reports/REPORT.md',body)
