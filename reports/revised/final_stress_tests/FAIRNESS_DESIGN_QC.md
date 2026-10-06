# Categorical UGR design gate

{
  "status": "stopped_design_qc",
  "reason": "categorical EV VIF exceeds 100",
  "rendered_runs": 1,
  "development_n": 176,
  "min_trials_per_cell": 3,
  "max_imbalance": 4.0,
  "max_task_ev_correlation": 0.95,
  "max_vif": 100.0,
  "max_condition_number": 100000000.0,
  "holdout_scored": false,
  "pilot_counts": {
    "social_unfair": 12,
    "social_fair": 12,
    "nonsocial_unfair": 12,
    "nonsocial_fair": 12,
    "endowment_high": 27,
    "endowment_low": 21
  },
  "pilot_design": {
    "n_volumes": 240,
    "active_columns": 37,
    "rank": 37,
    "condition_number": 26452797.77772735,
    "max_task_ev_correlation": 0.7859117712008151,
    "task_ev_vif": [
      12927964934730.42,
      13088579964666.553,
      12994260764696.406,
      13922467546267.258,
      19180487512094.336,
      19156651349626.04
    ],
    "relative_contrast_efficiency": [
      7.735169495343719e-14,
      7.640248237009386e-14,
      7.695705189454566e-14,
      7.182634807204914e-14,
      4.033974090935341e-09,
      1.2492910180824766e-10,
      3.6797552797363874e-10,
      2.6990248482264524e-07
    ],
    "failures": [
      "categorical EV VIF exceeds 100"
    ]
  }
}

First eligible run rendered and its actual convolved FEAT matrix checked before fitting. All runs must pass the same fixed thresholds; no performance-guided redefinition. The primary categorical regressors begin at offer onset and end at canonical choice-feedback end. Separate pre-offer, response/RT and missed-trial regressors retained. Relative efficiencies are design diagnostics, not power estimates.
