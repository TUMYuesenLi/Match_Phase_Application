"""Describe team phase composition, merged transitions and phase success."""
import numpy as np
import pandas as pd
from settings import DATA, PHASES, output_folder

out = output_folder("case_study")
selected_match = "DFL-MAT-0027DF"
teams = {"BVB": "Borussia Dortmund", "B04": "Bayer 04 Leverkusen"}
phases = pd.read_csv(DATA / "phase_occurrences.csv")
info = pd.read_csv(DATA / "match_metadata.csv")
team_matches = phases[["Match_id", "Episode", "Team"]].drop_duplicates()
team_matches = team_matches.merge(
    info[["Match_id", "Season"]], on="Match_id", validate="many_to_one"
)
time = pd.read_csv(DATA / "phase_time_counts.csv").rename(
    columns={"match": "Match_id", "category": "phase"}
)
time = time.merge(team_matches, on=["Match_id", "Episode"], validate="many_to_one")
composition = []
for team_id, team in teams.items():
    matches = time.loc[time.Team.eq(team) & time.Season.eq("2016-2017")]
    for profile in ["Selected match", "General"]:
        group = (
            matches.loc[matches.Match_id.str.startswith(selected_match)]
            if profile == "Selected match"
            else matches
        )
        # Composition reference is the MEAN of team-match shares, including this match.
        for phase, rows in group.groupby("phase"):
            composition.append(
                {
                    "team_id": team_id,
                    "team": team,
                    "profile": profile,
                    "phase": phase,
                    "phase_proportion": rows.distribution.mean(),
                }
            )
pd.DataFrame(composition).to_csv(out / "phase_composition.csv", index=False)

raw = pd.read_csv(DATA / "case_study_formal_occurrences.csv").rename(
    columns={
        "Phase_pred_filtered_aim_filtered_fixed": "Phase",
        "Phase_pred_filtered_aim_filtered_fixed_indices": "formal_index",
    }
)
raw["successful_frames"] = np.rint(raw.n_frames * raw.success_frame_fraction).astype(
    int
)
segments = []
for (team, match), rows in raw.groupby(["team_id", "match_code"], sort=False):
    rows = rows.sort_values("start_frame").copy()
    adjacent = (
        rows.Phase.eq(rows.Phase.shift())
        & rows.Episode.eq(rows.Episode.shift())
        & rows.start_frame.sub(rows.end_frame.shift()).eq(1)
    )
    rows["merged_segment_id"] = (~adjacent).cumsum()
    for segment_id, group in rows.groupby("merged_segment_id", sort=False):
        first, last = group.iloc[0], group.iloc[-1]
        row = {
            "team_id": team,
            "match_code": match,
            "Phase": first.Phase,
            "merged_segment_id": segment_id,
            "start_frame": first.start_frame,
            "end_frame": last.end_frame,
            "n_frames": group.n_frames.sum(),
            "successful_frames": group.successful_frames.sum(),
            "raw_destination": last.raw_destination,
            "n_formal_occurrences": len(group),
        }
        row["success_frame_fraction"] = row["successful_frames"] / row["n_frames"]
        row["is_successful"] = int(np.rint(row["success_frame_fraction"]))
        segments.append(row)
segments = pd.DataFrame(segments)
rates = (
    segments.groupby(["team_id", "match_code", "Phase"])
    .agg(
        n_phase_segments=("is_successful", "size"),
        n_successful=("is_successful", "sum"),
        success_rate=("is_successful", "mean"),
    )
    .reset_index()
    .rename(columns={"Phase": "phase"})
)
rates.to_csv(out / "merged_match_phase_success_rates.csv", index=False)
rates.loc[rates.match_code.eq(selected_match)].to_csv(
    out / "selected_match_phase_success.csv", index=False
)

categories = {
    "Build Up": {
        "Progression+": ["Progression", "Sustained Threat", "Finishing"],
        "Maintenance": ["Maintenance"],
        "Interrupted": ["Interrupted, possession retained"],
        "Possession switch": ["Possession switched / sequence ended"],
    },
    "Sustained Threat": {
        "Finishing": ["Finishing"],
        "Backward": ["Progression", "Build Up", "Maintenance"],
        "Interrupted": ["Interrupted, possession retained"],
        "Possession switch": ["Possession switched / sequence ended"],
    },
}
transitions, references, scope = [], [], []
for team in teams:
    for profile in ["Selected match", "General"]:
        group = segments.loc[segments.team_id.eq(team)]
        use = (
            group.match_code.eq(selected_match)
            if profile == "Selected match"
            else ~group.match_code.eq(selected_match)
        )
        group = group.loc[use]
        scope.extend(
            {"team_id": team, "profile": profile, "match_code": m}
            for m in group.match_code.unique()
        )
        for origin, mapping in categories.items():
            origin_rows = group.loc[group.Phase.eq(origin)]
            for order, (category, destinations) in enumerate(mapping.items()):
                count = origin_rows.raw_destination.isin(destinations).sum()
                transitions.append(
                    {
                        "team_id": team,
                        "profile": profile,
                        "origin_phase": origin,
                        "category_order": order,
                        "transfer_category": category,
                        "count": count,
                        "total_origin_occurrences": len(origin_rows),
                        "proportion": count / len(origin_rows),
                        "n_reference_matches": group.match_code.nunique(),
                    }
                )
        if profile == "General":
            for phase in PHASES:
                phase_rows = group.loc[group.Phase.eq(phase)]
                per_match = rates.loc[
                    rates.team_id.eq(team)
                    & rates.phase.eq(phase)
                    & ~rates.match_code.eq(selected_match)
                ]
                references.append(
                    {
                        "team_id": team,
                        "Phase": phase,
                        "n_reference_matches": group.match_code.nunique(),
                        "total_phase_segments": len(phase_rows),
                        "total_successful_segments": phase_rows.is_successful.sum(),
                        "Mean_success_rate": phase_rows.is_successful.mean(),
                        "SD": per_match.success_rate.std(ddof=1),
                        "reference_excludes_selected_match": True,
                    }
                )
pd.DataFrame(transitions).to_csv(out / "phase_transition_summary.csv", index=False)
pd.DataFrame(references).to_csv(
    out / "team_general_phase_success_references.csv", index=False
)
pd.DataFrame(scope).to_csv(out / "reference_match_scope.csv", index=False)
print(
    "O5:",
    len(raw),
    "formal occurrences merged into",
    len(segments),
    "segments.",
    flush=True,
)
