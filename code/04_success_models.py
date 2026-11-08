"""Fit separate adjusted logistic models of phase success.

Keep informative matches and model-stability checks because they determine
the analysis sample and the p-values entering BH correction."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from behaviour_data import prepare_behaviour_data
from settings import output_folder
from stat_output import anonymize_summary

out = output_folder("success_models")
native = (out / "model_output.txt").open("w", encoding="utf-8")
native.write("Native statsmodels GLM summaries from the reproducibility run\n")
native.write(
    "All fitted terms are included. Full-precision estimates are in model_coefficients.csv.\n\n"
)
long, scope = prepare_behaviour_data()
rows, diagnostics = [], []
coefficients, fit_statistics = [], []
grouping = ["Analysis_id", "Indicator", "Indicator_label", "Stage", "Phase"]
for keys, group in long.groupby(grouping, sort=False):
    common = dict(zip(grouping, keys))
    predictor = "Exposure" if common["Stage"] == "Presence" else "z_exposure"
    # A match must contain both outcomes in this indicator's analysis sample.
    variation = group.groupby("Match_id").success.nunique()
    matches = variation.loc[variation.eq(2)].index
    sample = group.loc[group.Match_id.isin(matches)].copy()
    formula = (
        f"success ~ {predictor} + "
        "C(Strength, Treatment(reference='Middle')) + z_log_duration + C(Match_id)"
    )
    with threadpool_limits(limits=1), warnings.catch_warnings(
        record=True
    ) as fit_warnings:
        warnings.simplefilter("always")
        model = smf.glm(formula, data=sample, family=sm.families.Binomial()).fit(
            maxiter=200
        )
        rank = np.linalg.matrix_rank(model.model.exog)
    warning_text = " | ".join(str(item.message) for item in fit_warnings)
    native.write("=" * 90 + "\n")
    native.write(f'{common["Analysis_id"]} | {common["Phase"]}\n{formula}\n')
    native.write(
        f"Input occurrences: {len(group)}; fitted occurrences: {len(sample)}\n"
    )
    native.write("Fit warnings: " + (warning_text or "None") + "\n")
    native.write(anonymize_summary(model.summary().as_text()) + "\n\n")
    intervals = model.conf_int(alpha=0.05)
    for term in model.params.index:
        coefficients.append(
            {
                **common,
                "term": term,
                "coefficient": model.params[term],
                "standard_error": model.bse[term],
                "z": model.tvalues[term],
                "p_raw": model.pvalues[term],
                "CI95_low_log_odds": intervals.loc[term, 0],
                "CI95_high_log_odds": intervals.loc[term, 1],
            }
        )
    fit_statistics.append(
        {
            **common,
            "n_observations": model.nobs,
            "n_parameters": model.model.exog.shape[1],
            "df_model": model.df_model,
            "df_residual": model.df_resid,
            "log_likelihood": model.llf,
            "deviance": model.deviance,
            "pearson_chi2": model.pearson_chi2,
            "aic": model.aic,
            "bic_log_likelihood": model.bic_llf,
            "iterations": model.fit_history["iteration"],
            "converged": model.converged,
            "fit_warnings": warning_text,
            "formula": formula,
        }
    )
    beta = model.params[predictor]
    se = model.bse[predictor]
    finite = (
        np.isfinite(model.params).all()
        and np.isfinite(model.bse).all()
        and np.isfinite(model.llf)
    )
    stable = (
        model.converged
        and rank == model.model.exog.shape[1]
        and finite
        and model.params.abs().max() <= 10
        and model.bse.max() <= 10
    )
    rows.append(
        {
            **common,
            "n_occurrences": len(sample),
            "n_matches": sample.Match_id.nunique(),
            "n_successful": sample.success.sum(),
            "log_OR": beta,
            "SE_log_OR": se,
            "OR": np.exp(beta),
            "CI95_low": np.exp(beta - 1.96 * se),
            "CI95_high": np.exp(beta + 1.96 * se),
            "p_raw": model.pvalues[predictor] if stable else np.nan,
            "Model_stable": stable,
            "formula": formula,
        }
    )
    diagnostics.append(
        {
            **common,
            "n_occurrences_all": len(group),
            "n_matches_all": group.Match_id.nunique(),
            "n_occurrences_informative_matches": len(sample),
            "n_informative_matches": sample.Match_id.nunique(),
            "converged": model.converged,
            "design_rank": rank,
            "n_parameters": model.model.exog.shape[1],
            "max_abs_coefficient": model.params.abs().max(),
            "max_standard_error": model.bse.max(),
            "stable": stable,
        }
    )
    print(
        f'O4: fitted {len(rows)}/{len(scope)}: {common["Indicator"]}, {common["Phase"]}',
        flush=True,
    )
results = pd.DataFrame(rows)
results["q_BH_across_scoped_phases"] = np.nan
for _, indices in results.groupby("Analysis_id").groups.items():
    valid = results.loc[indices, "p_raw"].dropna()
    results.loc[valid.index, "q_BH_across_scoped_phases"] = multipletests(
        valid, method="fdr_bh"
    )[1]
results["Significant"] = results.Model_stable & results.q_BH_across_scoped_phases.lt(
    0.05
)
results["Supported"] = results.Significant & (results.OR.ge(1.44) | results.OR.le(0.70))
results.to_csv(out / "adjusted_logistic_results.csv", index=False)
pd.DataFrame(diagnostics).to_csv(out / "model_diagnostics.csv", index=False)
pd.DataFrame(coefficients).to_csv(
    out / "model_coefficients.csv", index=False, float_format="%.17g"
)
pd.DataFrame(fit_statistics).to_csv(
    out / "model_fit_statistics.csv", index=False, float_format="%.17g"
)
native.close()
print(
    "O4: saved",
    len(results),
    "models;",
    results.Supported.sum(),
    "supported associations.",
    flush=True,
)
