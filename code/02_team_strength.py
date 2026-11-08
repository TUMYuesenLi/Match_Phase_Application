"""Compare phase success across team-strength groups and describe the sample."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from settings import DATA, PHASES, output_folder
from stat_output import write_native_result

STRENGTHS = ["Top 4", "Middle", "Bottom 4"]

out = output_folder("team_strength")
native = (out / "test_output.txt").open("w", encoding="utf-8")
native.write("Native SciPy outputs: team-strength comparisons\n\n")
phases = pd.read_csv(DATA / "phase_occurrences.csv")
phases["is_successful"] = np.rint(phases.success_frame_fraction).astype(int)
keys = ["Match_id", "Episode", "Phase", "Team", "Venue", "Outcome", "Rank", "Strength"]
rates = (
    phases.groupby(keys)
    .agg(
        n_phase_segments=("is_successful", "size"),
        n_successful=("is_successful", "sum"),
    )
    .reset_index()
)
rates["success_rate"] = rates.n_successful / rates.n_phase_segments
rates.to_csv(out / "match_team_phase_success_rates.csv", index=False)

descriptive, omnibus, pairwise, baseline = [], [], [], []
for phase in PHASES:
    group = rates.loc[rates.Phase.eq(phase)]
    baseline.append(
        {
            "Phase": phase,
            "weighted_success_rate": group.n_successful.sum()
            / group.n_phase_segments.sum(),
            "mean_success_rate": group.success_rate.mean(),
            "sd_success_rate": group.success_rate.std(),
            "total_phase_segments": group.n_phase_segments.sum(),
            "total_successful_segments": group.n_successful.sum(),
        }
    )
    arrays = []
    for strength in STRENGTHS:
        selected = group.loc[group.Strength.eq(strength)]
        values = selected.success_rate.to_numpy()
        arrays.append(values)
        descriptive.append(
            {
                "Comparison": "Strength",
                "Phase": phase,
                "Group": strength,
                "n_match_team_values": len(values),
                "total_phase_segments": selected.n_phase_segments.sum(),
                "total_successful_segments": selected.n_successful.sum(),
                "Mean_success_rate": values.mean(),
                "SD": values.std(ddof=1),
                "SEM": stats.sem(values),
                "Median": np.median(values),
                "IQR": np.quantile(values, 0.75) - np.quantile(values, 0.25),
            }
        )
    test = stats.kruskal(*arrays)
    write_native_result(
        native, phase + ": Kruskal-Wallis, Top 4 / Middle / Bottom 4", test
    )
    n = sum(len(a) for a in arrays)
    omnibus.append(
        {
            "Comparison": "Strength",
            "Phase": phase,
            "Statistic": test.statistic,
            "p_raw": test.pvalue,
            "n_values": n,
            "Effect": max(0, (test.statistic - 3 + 1) / (n - 3)),
        }
    )
    first_pair = len(pairwise)
    for a, b in combinations(range(3), 2):
        x, y = arrays[a], arrays[b]
        test = stats.mannwhitneyu(
            x, y, alternative="two-sided", method="asymptotic", use_continuity=True
        )
        write_native_result(native, f"{phase}: {STRENGTHS[a]} vs {STRENGTHS[b]}", test)
        # Match the original tie- and continuity-corrected Z²/N effect size.
        nx, ny = len(x), len(y)
        total = nx + ny
        ties = np.unique(np.concatenate([x, y]), return_counts=True)[1].astype(float)
        variance = (
            nx
            * ny
            / 12
            * (total + 1 - (ties ** 3 - ties).sum() / (total * (total - 1)))
        )
        z = max(0, abs(test.statistic - nx * ny / 2) - 0.5) / np.sqrt(variance)
        pairwise.append(
            {
                "Comparison": "Strength",
                "Phase": phase,
                "Contrast": f"{STRENGTHS[a]} - {STRENGTHS[b]}",
                "n_a": nx,
                "n_b": ny,
                "Statistic": test.statistic,
                "p_raw": test.pvalue,
                "Effect": z * z / total,
            }
        )
    holm = multipletests([r["p_raw"] for r in pairwise[first_pair:]], method="holm")[1]
    for row, p in zip(pairwise[first_pair:], holm):
        row["p_holm"] = p
omnibus = pd.DataFrame(omnibus)
omnibus["q_FDR"] = multipletests(omnibus.p_raw, method="fdr_bh")[1]
pairwise = pd.DataFrame(pairwise)
pairwise["q_pair_FDR"] = multipletests(pairwise.p_raw, method="fdr_bh")[1]
omnibus.to_csv(out / "primary_tests.csv", index=False)
pairwise.to_csv(out / "strength_pairwise_tests.csv", index=False)
pd.DataFrame(descriptive).to_csv(out / "descriptive_statistics.csv", index=False)
pd.DataFrame(baseline).to_csv(out / "general_phase_baseline.csv", index=False)

# Sample summary: each team-match counts once, even if a phase is absent.
metadata = phases[["Match_id", "Episode", "Team", "Rank", "Strength"]].drop_duplicates()
info = pd.read_csv(DATA / "match_metadata.csv")
metadata = metadata.merge(
    info[["Match_id", "Season"]], on="Match_id", validate="many_to_one"
)
home = info[["Season", "Home", "Home_rank"]].rename(
    columns={"Home": "Team", "Home_rank": "Rank"}
)
away = info[["Season", "Away", "Away_rank"]].rename(
    columns={"Away": "Team", "Away_rank": "Rank"}
)
# Match the full club names printed in the sample summary.
sample = pd.concat([home, away])
sample["Team"] = sample.Team.replace(
    {"Hertha Berlin": "Hertha BSC", "Borussia M'gladbach": "Borussia Mönchengladbach"}
)
sample.groupby(["Season", "Team", "Rank"]).size().rename(
    "Matches"
).reset_index().to_csv(out / "sample_by_team_season.csv", index=False)

# Time and count summaries are descriptive only. Explicit zero counts retain all 292 team-matches.
time = pd.read_csv(DATA / "phase_time_counts.csv").rename(
    columns={"match": "Match_id", "category": "Phase"}
)
time = time.merge(
    metadata[["Match_id", "Episode", "Strength"]],
    on=["Match_id", "Episode"],
    validate="many_to_one",
)
counts = (
    phases.groupby(["Match_id", "Episode", "Phase"])
    .size()
    .rename("occurrences")
    .reset_index()
)
time = time.merge(counts, on=["Match_id", "Episode", "Phase"], how="left")
time["occurrences"] = time.occurrences.fillna(0).astype(int)
time["time_share_pct"] = 100 * time.n_frames / time.total_frames
time.to_csv(out / "team_match_phase_time_and_counts.csv", index=False)
time.groupby(["Phase", "Strength"]).agg(
    n_team_matches=("Match_id", "size"),
    time_share_mean=("time_share_pct", "mean"),
    time_share_sd=("time_share_pct", "std"),
    occurrence_mean=("occurrences", "mean"),
    occurrence_sd=("occurrences", "std"),
).reset_index().to_csv(out / "phase_time_and_counts_by_strength.csv", index=False)
print(
    "O2: six strength tests, 18 contrasts and supplementary summaries saved.",
    flush=True,
)
native.close()
