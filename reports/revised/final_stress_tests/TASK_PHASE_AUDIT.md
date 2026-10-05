# Task-phase audit: final development stress tests

Source-code review completed 2026-10-04 before new fits. No private raw PsychoPy logs or protected participant images were opened. Exact source revisions and SHA256 values are recorded in `config/final_stress_sources.json`; production verifies the analysis/conversion files before execution. This establishes the behavior of the inspected presentation code, not unrecorded differences in every historical deployed copy.

## Shared Reward

Presentation: [SRPAL-DataInBrief/stimuli/Scan-Card_Guessing_Game/card_guessing_game.py](https://github.com/DVS-Lab/SRPAL-DataInBrief/blob/990ae8485aa7329e3bb765152e7fbe299d940f76/stimuli/Scan-Card_Guessing_Game/card_guessing_game.py#L209).

Lines 209–264 implement the high/low decision: `pictureStim.draw()` and `nameStim.draw()` occur at 224–225. After response these calls are commented out at 251–252. During outcomes (292–352), the partner-image/name draw calls are commented out at 299–300; the card, numerical value, valence symbol and color are drawn at 298 and 341–345. Thus the inspected outcome display contains neither partner picture nor name.

Authoritative analysis: `rf1-sra-sharedreward/templates/L1_task-sharedreward_model-1_type-act.fsf`. EVs 1–9 are C/F/S punishment, reward and neutral outcomes; 10–11 model missed decision/outcome; 12–14 are F_dec, S_dec, C_dec. The target single-EV COPEs are 1=C_pun, 2=C_rew, 3=F_pun, 4=F_rew, 5=S_pun, 6=S_rew. `code/gen3colfiles.sh` and `code/BIDSto3col.sh` preserve canonical trial-type onset/duration. `rf1-sra-linux2/code/convert_behavior.py:655–746` writes face/non-face decision events from decision onset for response-time duration and separate outcome events from outcome onset to offset.

Production sensitivity paths are `/ZPOOL/data/projects/rf1-sra-sharedreward/derivatives/fsl/sub-<ID>/ses-01/L2_task-sharedreward_ses-01_model-1_type-act_smTo-6.gfeat/cope<K>.feat/stats/cope1.nii.gz`. The completed October 5 activation audit (`logs/records/phase-resolved-20261005-144746`, source commit `d5331d38c298023eeaf094e6e83ee0df65434ac2`) verifies 655 runs / 346 subject-sessions: 309 two-run fixed-effects outputs and 37 explicitly recorded L1 passthroughs. The classifier consumes that pinned manifest, validates the recorded source runs and completed designs, and applies existing QC to the contributing runs only. An absent L2 never causes an inferred fallback. A recorded passthrough instead uses `L1_task-sharedreward_ses-01_model-1_type-act_run-R_smTo-6.feat/stats/copeK.nii.gz`. Empty neutral/miss EVs are permitted as explicit shape-10 columns; substantive outcome and decision events remain required. COPEs 1–6 and their vectors are unchanged. The existing aging full-trial maps and frozen signatures remain unchanged.

## Trust

Presentation: [SRPAL-DataInBrief/stimuli/Scan-Investment_Game/investment_game.py](https://github.com/DVS-Lab/SRPAL-DataInBrief/blob/990ae8485aa7329e3bb765152e7fbe299d940f76/stimuli/Scan-Investment_Game/investment_game.py#L280).

Decision draws picture, sharing text and investment options (284–292, 324–327). Outcome text is constructed from `"{} kept money"` / `"{} shared money"` (117, 394); the text and valence symbol are drawn at 408–409, without `pictureStim.draw()`. Positive/negative colors are lime/dark red (399–406). A zero investment produces fixation (386–392); missed responses are separate. The modeled reciprocation/defection outcomes refer to invested trials, not those fixation events.

`rf1-sra-trust/templates/L1_task-trust_model-1_type-act.fsf` separates C/F/S decision EVs 1–3 from outcome COPEs 4=C_def, 5=C_rec, 6=F_def, 7=F_rec, 8=S_def, 9=S_rec. `code/gen3colfiles.sh` requires choice and outcome categories separately; `rf1-sra-linux2/code/convert_behavior.py` writes outcomes using logged outcome onset/offset. Partner text is present, so absence of faces does not remove identity/semantic differences.

## Ultimatum Game

Presentation: [SRPAL-DataInBrief/stimuli/Scan-Lets_Make_A_Deal/lets_make_a_deal-scan.py](https://github.com/DVS-Lab/SRPAL-DataInBrief/blob/990ae8485aa7329e3bb765152e7fbe299d940f76/stimuli/Scan-Lets_Make_A_Deal/lets_make_a_deal-scan.py#L218).

Social trials select trial-varying face files (218–233). Nonsocial trials select the computer cue. `cueStim.draw()` continues through the endowment and offer/choice display (244, 277, 320; nonsocial 346–356, 389, 432); the offer and endowment text are also displayed. The actual offer phase therefore still includes a face/computer cue.

Authoritative `rf1-sra-ugr/code/gen_model3_evs.py:collapse_trials` defines broad onset as partner-cue onset and the endpoint as the end of canonical `choice_feedback`. `build_ev_rows` uses that broad duration for all four constants. Its RT constant and mean-centered RT pmod are impulses at response onset. Template `templates/L1_task-ugr_model-3_type-act.fsf` maps constants to COPE 1=nonsocial_high, 3=nonsocial_low, 5=social_high, 7=social_low. Model 3 uses 5 mm smoothing, no temporal high-pass filtering, double-gamma HRF, no derivatives, and the authoritative TEDANA-plus-confounds file. These settings are preserved downstream.

Canonical BIDS in `rf1-sra-linux2/code/convert_behavior.py:918–1035` stores a **dollar offer**, obtained from the nonzero option. Offer category is not response/acceptance. The user approved nominal percentages with rounded dollars: at $16, unfair $1/$2 and fair $4/$8; at $32, unfair $2/$3 and fair $8/$16. Unexpected/ambiguous offers stop the categorical branch. The archived timing-generator MATLAB file is not used to infer categories; it describes a different endowment schedule.

The new categorical model starts at canonical **decision/offer onset** and ends at canonical choice-feedback end. It excludes the preceding cue/endowment epoch from fairness regressors, models that epoch separately by sociality/endowment, and preserves response/RT and missed-trial nuisance events. It never replaces model 3. UGR social/computer transfer remains a **visually confounded third-task generalization probe**, including when training uses outcome-only estimates.

## Social Doors and interpretive hierarchy

`SRPAL-DataInBrief/stimuli/Scan-Social_Doors/social_doors.py` chooses stimulus sets at 93–95 and draws two choice pictures at 150–151 and 162–180. Faces and doors are distinct decision stimuli. Doors remains a boundary paradigm, not a clean perceptual control; no new Doors fit is needed for this package.

The primary new evidence is phase-resolved Shared Reward outcome ↔ Trust outcome transfer in people absent from all training domains. Both inspected target epochs lack concurrent faces. Success would weigh against a simple concurrent face/nonface account, but cannot remove hemodynamic carryover, familiarity, identity, names/text, semantic context or all perceptual alternatives.

The exact prior Harvard-Oxford occipital exclusion is authorized **only as descriptive sensitivity**. It cannot select a model, is not a persistence/success criterion, and survival cannot establish abstract social coding. Frozen full-DEV applications on new development maps involve training-participant overlap and are labeled separately from OOF generalization. The social norm-violation family is exploratory and cannot modify either candidate or its validation plan.

holdout_scored = False
