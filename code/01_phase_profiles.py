"""Descriptive phase distributions, durations and match-level success fractions."""
import pandas as pd
from settings import DATA, PHASES, output_folder

out = output_folder("phase_profiles")
phases = pd.read_csv(DATA / "phase_occurrences.csv")
counts = pd.read_csv(DATA / "phase_time_counts.csv")

# Team-match time shares have a common denominator across all six phases.
counts["distribution"] = counts.n_frames / counts.total_frames

# Descriptive match-level success calculation: mean successful-frame
# fraction per occurrence, then mean across occurrences within a MATCH.
# Script 02 uses rounded binary labels and team-match rates instead.
success = (
    phases.groupby(["Match_id", "Phase"]).success_frame_fraction.mean().reset_index()
)
success.columns = ["match", "category", "success_rate"]
success.to_csv(out / "phase_success_per_match.csv", index=False)
summary = []
for phase in PHASES:
    shares = counts.loc[counts.category.eq(phase), "distribution"]
    durations = phases.loc[phases.Phase.eq(phase), "duration_s"]
    rates = success.loc[success.category.eq(phase), "success_rate"]
    summary.append(
        {
            "category": phase,
            "distribution_mean": shares.mean(),
            "distribution_std": shares.std(),
            "n_anonymous_teams": len(shares),
            "duration_mean_s": durations.mean(),
            "duration_std_s": durations.std(),
            "n_occurrences": len(durations),
            "success_rate_mean": rates.mean(),
            "success_rate_std": rates.std(),
            "n_success_matches": len(rates),
        }
    )
pd.DataFrame(summary).to_csv(out / "phase_summary.csv", index=False)

print("Phase profiles saved.", flush=True)
