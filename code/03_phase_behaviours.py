"""Compare behavioural indicators across phases using rank and presence tests.

Continuous: Kruskal-Wallis and Mann-Whitney. Binary: chi-square and Fisher.
Effect sizes describe explained variation in average ranks."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from settings import DATA, PHASES, output_folder
from stat_output import write_native_result


def rank_eta(data, factor):
    ranks = data.Exposure.rank(method="average")
    grand_mean = ranks.mean()
    work = pd.DataFrame({"group": data[factor], "rank": ranks})
    between = 0
    within = 0
    for _, group in work.groupby("group"):
        between += len(group) * (group["rank"].mean() - grand_mean) ** 2
        within += ((group["rank"] - group["rank"].mean()) ** 2).sum()
    return between / (between + within)


out = output_folder("phase_behaviours")
native = (out / "test_output.txt").open("w", encoding="utf-8")
native.write("Native SciPy outputs: between-phase behaviour comparisons\n\n")
scope = pd.read_csv(DATA / "indicator_scope.csv")
# Use the original analysis values at full precision. Rebuilding values from
# a separately saved wide CSV can split floating-point ties and change ranks.
long = pd.read_csv(DATA / "phase_profile_values.csv.gz", float_precision="round_trip")
labels = (
    scope[["Indicator", "Domain", "Indicator_label"]]
    .drop_duplicates()
    .set_index("Indicator")
)
descriptives, omnibus, pairwise = [], [], []
for (indicator, stage), group in long.groupby(["Indicator", "Stage"], sort=False):
    label = labels.loc[indicator]
    common = {
        "Domain": label.Domain,
        "Indicator": indicator,
        "Indicator_label": label.Indicator_label,
        "Stage": stage,
    }
    available = [p for p in PHASES if group.Phase.eq(p).any()]
    arrays = [group.loc[group.Phase.eq(p), "Exposure"].to_numpy() for p in available]
    for phase, values in zip(available, arrays):
        ci_low, ci_high = np.nan, np.nan
        if stage == "Presence":
            z = stats.norm.ppf(0.975)
            n = len(values)
            proportion = values.mean()
            denominator = 1 + z * z / n
            centre = (proportion + z * z / (2 * n)) / denominator
            half = (
                z
                * np.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n))
                / denominator
            )
            ci_low, ci_high = centre - half, centre + half
        descriptives.append(
            {
                **common,
                "Phase": phase,
                "n_occurrences": len(values),
                "n_matches": group.loc[group.Phase.eq(phase), "Match_id"].nunique(),
                "Mean_or_prevalence": values.mean(),
                "SD": values.std(ddof=1),
                "Median": np.median(values),
                "Q1": np.quantile(values, 0.25),
                "Q3": np.quantile(values, 0.75),
                "Wilson_CI95_low": ci_low,
                "Wilson_CI95_high": ci_high,
                "Pooled_mean": group.Exposure.mean(),
                "Pooled_SD": group.Exposure.std(ddof=0),
            }
        )
    if stage == "Presence":
        table = pd.crosstab(group.Phase, group.Exposure).reindex(
            index=available, columns=[0.0, 1.0], fill_value=0
        )
        test = stats.chi2_contingency(table.to_numpy(), correction=False)
        statistic, p, df, expected = test
        native.write(
            "Observed contingency table for "
            + indicator
            + "\n"
            + table.to_string()
            + "\n"
        )
        native.write(
            "Expected frequencies\n" + np.array2string(expected, precision=17) + "\n"
        )
        test_name = "Pearson chi-square"
    else:
        test = stats.kruskal(*arrays)
        statistic, p, df = test.statistic, test.pvalue, len(available) - 1
        test_name = "Kruskal-Wallis"
    write_native_result(native, indicator + ": " + test_name, test)
    omnibus.append(
        {
            **common,
            "Test": test_name,
            "Statistic": statistic,
            "df": df,
            "p_raw": p,
            "n_phases": len(available),
            "n_occurrences": len(group),
            "n_matches": group.Match_id.nunique(),
            "partial_eta_squared": rank_eta(group, "Phase"),
        }
    )
    first = len(pairwise)
    for a, b in combinations(available, 2):
        pair = group.loc[group.Phase.isin([a, b])]
        x = pair.loc[pair.Phase.eq(a), "Exposure"].to_numpy()
        y = pair.loc[pair.Phase.eq(b), "Exposure"].to_numpy()
        if stage == "Presence":
            table = pd.crosstab(pair.Phase, pair.Exposure).reindex(
                index=[a, b], columns=[0.0, 1.0], fill_value=0
            )
            test = stats.fisher_exact(table.to_numpy())
            statistic, p = test
            native.write("Observed contingency table\n" + table.to_string() + "\n")
            test_name = "Fisher exact"
        else:
            test = stats.mannwhitneyu(
                x, y, alternative="two-sided", method="asymptotic", use_continuity=True
            )
            statistic, p = test.statistic, test.pvalue
            test_name = "Mann-Whitney U"
        write_native_result(native, f"{indicator}: {a} vs {b}, {test_name}", test)
        pairwise.append(
            {
                **common,
                "Phase_1": a,
                "Phase_2": b,
                "Contrast": f"{a} vs {b}",
                "n_phase_1": len(x),
                "n_phase_2": len(y),
                "Mean_or_prevalence_phase_1": x.mean(),
                "Mean_or_prevalence_phase_2": y.mean(),
                "Test": test_name,
                "Statistic": statistic,
                "p_raw": p,
                "partial_eta_squared": rank_eta(pair, "Phase"),
            }
        )
    # Original pairwise family: phase contrasts WITHIN each indicator.
    raw = [r["p_raw"] for r in pairwise[first:]]
    holm = multipletests(raw, method="holm")[1]
    bh = multipletests(raw, method="fdr_bh")[1]
    for row, h, q in zip(pairwise[first:], holm, bh):
        row["p_Holm_within_indicator"] = h
        row["q_BH_within_indicator"] = q
omnibus = pd.DataFrame(omnibus)
omnibus["q_BH_across_indicators"] = multipletests(omnibus.p_raw, method="fdr_bh")[1]
omnibus["Supported"] = omnibus.q_BH_across_indicators.lt(
    0.05
) & omnibus.partial_eta_squared.gt(0.01)
pairwise = pd.DataFrame(pairwise)
pairwise["Supported"] = pairwise.q_BH_within_indicator.lt(
    0.05
) & pairwise.partial_eta_squared.gt(0.01)
descriptives = pd.DataFrame(descriptives)
descriptives.to_csv(out / "phase_indicator_descriptive_statistics.csv", index=False)
omnibus.to_csv(out / "phase_indicator_omnibus_tests.csv", index=False)
pairwise.to_csv(out / "phase_indicator_pairwise_tests.csv", index=False)
print(
    "O3:",
    len(omnibus),
    "indicator tests and",
    len(pairwise),
    "contrasts saved.",
    flush=True,
)
native.close()
