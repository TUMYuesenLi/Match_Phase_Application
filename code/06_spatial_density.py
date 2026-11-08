"""Estimate spatial density from aggregate weighted bins, without plotting.

The input contains summed phase-balanced weights, not frame coordinates.
Each phase with valid on-pitch observations contributes total input weight 1.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from settings import DATA, PHASES, output_folder

out = output_folder("spatial_density")
bins = pd.read_csv(DATA / "spatial_weighted_histogram.csv")
grid_rows, contour_rows, summaries = [], [], []
for phase in PHASES:
    group = bins.loc[bins.Phase.eq(phase)]
    histogram = group.pivot(
        index="x_bin", columns="y_bin", values="weighted_count"
    ).to_numpy()
    density = gaussian_filter(histogram, sigma=1.45, mode="nearest").T
    density = density / density.max()
    x = (np.arange(70) + 0.5) * 105 / 70
    y = (np.arange(46) + 0.5) * 68 / 46
    for j in range(46):
        for i in range(70):
            grid_rows.append(
                {"Phase": phase, "x": x[i], "y": y[j], "density": density[j, i]}
            )
    ordered = np.sort(density[density > 0])[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    for mass in [0.50, 0.80, 0.95]:
        threshold = ordered[np.searchsorted(cumulative, mass)]
        contour_rows.append(
            {"Phase": phase, "probability_mass": mass, "density_threshold": threshold}
        )
    summaries.append(
        {
            "Phase": phase,
            "n_phase_occurrences_with_valid_positions": int(round(histogram.sum())),
            "weighted_bin_sum": histogram.sum(),
            "x_bins": 70,
            "y_bins": 46,
            "pitch_length_m": 105,
            "pitch_width_m": 68,
            "gaussian_sigma_bins": 1.45,
            "boundary_mode": "nearest",
            "density_normalization": "phase maximum equals 1",
        }
    )
pd.DataFrame(grid_rows).to_csv(out / "spatial_density_grid.csv", index=False)
pd.DataFrame(contour_rows).to_csv(out / "spatial_density_contours.csv", index=False)
pd.DataFrame(summaries).to_csv(out / "spatial_phase_summary.csv", index=False)
print("Spatial estimates saved; no figures generated.", flush=True)
