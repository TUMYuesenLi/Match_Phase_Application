"""Convert the released phase-level indicator table into analysis rows.

Functions here only prepare data. Scripts 03 and 04 perform the statistics.
"""
import numpy as np
import pandas as pd
from settings import DATA


def prepare_behaviour_data():
    data = pd.read_csv(DATA / "behaviour_occurrences.csv", float_precision="round_trip")
    scope = pd.read_csv(DATA / "indicator_scope.csv")
    # No carry means zero carry-time share; missing carry distance remains missing.
    carry_time = data.total_carry_duration_s.where(data.n_carries.gt(0), 0)
    data["carry_time_share"] = (carry_time / data.duration_s).clip(0, 1)
    # No classifiable passes means a missing high-pass ratio, not a zero.
    data["high_pass_ratio_zero_inclusive"] = data.high_pass_proportion.where(
        data.n_height_classified.gt(0)
    )
    data["switch_play"] = data.n_switch_passes.fillna(0).gt(0).astype(float)
    data["cross"] = data.n_crosses.gt(0).astype(float)
    parts = []
    for row in scope.itertuples(index=False):
        group = data.loc[data.Phase.eq(row.Phase)].dropna(subset=[row.Indicator]).copy()
        part = group[
            [
                "Match_id",
                "phase_occurrence_id",
                "Phase",
                "Phase_result",
                "success",
                "duration_s",
                "Strength",
            ]
        ].copy()
        part["Indicator"] = row.Indicator
        part["Indicator_label"] = row.Indicator_label
        part["Stage"] = row.Stage
        part["Exposure"] = group[row.Indicator].to_numpy()
        part["Analysis_id"] = row.Indicator + "__" + row.Stage
        # Standardise BEFORE retaining informative matches. This ordering is
        # part of the original analysis and determines the OR's one-SD unit.
        log_duration = np.log(part.duration_s)
        part["z_log_duration"] = (
            log_duration - log_duration.mean()
        ) / log_duration.std(ddof=0)
        part["z_exposure"] = np.nan
        if row.Stage != "Presence":
            part["z_exposure"] = (
                part.Exposure - part.Exposure.mean()
            ) / part.Exposure.std(ddof=0)
        parts.append(part)
    return pd.concat(parts, ignore_index=True), scope
