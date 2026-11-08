# Match phase statistics

Anonymous review materials for a study of football match phases. The package
reproduces phase profiles, team-strength comparisons, behavioural comparisons,
adjusted success models, a descriptive case study, and aggregate spatial estimates.
It generates no figures and requires no plotting libraries.

## Run

Use Python 3.9. From this directory:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

All inputs are included. Outputs are written to `results/` and
`statistical_output.txt`; rerunning replaces these generated files. Library
versions are pinned in `requirements.txt`, and the full output records the
actual environment. Individual scripts can also be run with
`python code/04_success_models.py`, for example.

## Inputs and relationships

| Input in `data/` | Role and identifying keys |
|---|---|
| `phase_occurrences.csv` | 62,032 phases from 146 matches; join by `Match_id` and `phase_occurrence_id` |
| `phase_time_counts.csv` | Classified frame counts and time shares; `match` corresponds to `Match_id`; `Episode` identifies the possession team; `category` is the phase name |
| `match_metadata.csv` | Match, season, competition, team and rank metadata; join by `Match_id` |
| `behaviour_occurrences.csv` | 60,701 phase-level behaviour records after the original duration filter; join by `Match_id` and `phase_occurrence_id` |
| `phase_profile_values.csv.gz` | Original full-precision behavioural comparison values; `Match_id`, `phase_occurrence_id`, `Indicator`, `Stage` and `Phase` identify a value |
| `indicator_scope.csv` | Applicable indicator–phase cells and labels; join by `Indicator`, `Stage` and `Phase` |
| `case_study_formal_occurrences.csv` | Formal phase boundaries, success fractions and successor categories for the case-study teams; match code and team identify each sequence |
| `spatial_weighted_histogram.csv` | Aggregate spatial weights; one row per `Phase`, `x_bin`, `y_bin` |

`data_dictionary.csv` defines all input fields. Empty numeric fields represent
missing values. They must not generally be replaced by zero. The separate
behavioural comparison file preserves floating-point ties; do not regenerate
it from rounded wide-table values.

## Data source and processing boundary

The research used Bundesliga tracking and event data. The upstream pipeline
identified tactical phases and calculated phase-level outcomes and indicators.
The released inputs start after that identification and aggregation. Raw player
trajectories, ball-coordinate time series, action-event records, and the
phase-identification model are not included. Their extraction cannot be rerun
from this package alone.

The phase sample contains 62,032 occurrences. Behaviour analyses preserve the
original exclusion of non-Finishing phases shorter than 0.5 seconds, leaving
60,701 occurrences; indicator-specific missing values are then excluded.
Descriptive match-level success fractions and team–match binary success rates
are different estimands and are preserved separately.

## Output guide

Scripts 01–05 generate the five core statistical result groups. Script 06
computes spatial estimates. Script 07 compiles `statistical_output.txt` from:

- Native SciPy test returns and observed/expected contingency tables.
- All 83 native statsmodels GLM summaries, including all fitted terms.
- Complete statistical result tables, including adjusted p-values and diagnostics.

`results/success_models/model_coefficients.csv` contains full-precision
coefficients, SEs, z statistics, p-values and confidence limits for every fitted
term, including strength, duration and match fixed effects. Coefficients and
their confidence limits are on the log-odds scale.
The main OR table retains the original intervals calculated with 1.96 × SE;
native summaries and the full coefficient table use statsmodels' exact normal
quantile. This small interval-rounding difference does not change the model fits.
`model_fit_statistics.csv` records sample sizes, likelihood, deviance, AIC,
log-likelihood BIC, iteration counts, convergence and captured fit warnings.
These are software outputs from a reproducibility run, not a recovered
historical console log. No test/model is omitted because of non-significance.

The main estimates are conditional associations from separate indicator models.
Continuous predictors and log duration are standardised before retaining
matches with both outcomes. BH families are six phase tests for team strength,
18 team-strength contrasts, 17 between-phase indicator tests, phase contrasts
within each indicator, and applicable phase models within each indicator.

## Spatial estimates

Off-pitch coordinates were excluded upstream. Each phase with valid coordinates
contributed total weight one across its retained frames. Only summed weights in
a 70 × 46 grid on a 105 × 68 m pitch are included. No frame coordinates are
released. There are 62,002 spatially represented phases; 30 have no valid
retained coordinates.

Script 06 applies a Gaussian filter with sigma 1.45 bins and boundary mode
`nearest`, normalises each phase density to a maximum of one, and derives
50/80/95% mass thresholds. This reproduces the weighted-histogram smoothing
estimator used for the spatial results. Outputs are numerical CSVs only.

## Rights and citation

**Licensing is pending. No MIT or other open-source/open-data license is
granted.** Code and data have separate rights considerations. See `LICENSE.md`
and `DATA_USE.md`; inclusion in this package does not establish permission to
redistribute third-party data. No public repository URL or DOI has been assigned
in these files. The package does not make the raw inputs available on request.

Author names, manuscript title and citation metadata are withheld in this
anonymous review copy. The non-anonymous master package retains that metadata.
See `ANONYMIZATION.md` for the scope of anonymization. Licensing remains pending.
