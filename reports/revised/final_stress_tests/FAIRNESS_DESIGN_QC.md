# Categorical UGR design gate

{
  "status": "preview_passed",
  "reason": null,
  "rendered_runs": 1,
  "development_n": 176,
  "min_trials_per_cell": 3,
  "max_imbalance": 4.0,
  "max_task_ev_correlation": 0.95,
  "max_vif": 100.0,
  "max_condition_number": 100000000.0,
  "holdout_scored": false,
  "pilot_counts": {
    "social_high_unfair": 6,
    "social_high_fair": 7,
    "social_low_unfair": 6,
    "social_low_fair": 5,
    "nonsocial_high_unfair": 8,
    "nonsocial_high_fair": 6,
    "nonsocial_low_unfair": 4,
    "nonsocial_low_fair": 6
  },
  "pilot_design": {
    "n_volumes": 240,
    "active_columns": 41,
    "rank": 41,
    "condition_number": 83.05918025388024,
    "max_task_ev_correlation": 0.1681259106252476,
    "task_ev_vif": [
      11.465068929778656,
      12.549325866008303,
      10.060179283395245,
      9.477479457436813,
      13.060449375515466,
      10.067931339033152,
      8.112885592134095,
      10.936800743602452
    ],
    "relative_contrast_efficiency": [
      0.050080072532400766,
      0.04924487635371572,
      0.0520269451840915,
      0.05113175396298427,
      0.5265798509079241,
      0.7545516268280429,
      0.6259340927685874
    ],
    "failures": []
  }
}

First eligible run rendered and its actual convolved FEAT matrix checked before fitting. All runs must pass the same fixed thresholds; no performance-guided redefinition. The primary categorical regressors begin at offer onset and end at canonical choice-feedback end. Separate pre-offer, response/RT and missed-trial regressors retained. Relative efficiencies are design diagnostics, not power estimates.
