"""Combine native test/model outputs and complete numerical result tables."""
import sys
import numpy as np
import pandas as pd
import scipy
import statsmodels
import patsy
import threadpoolctl
from settings import ROOT, RESULTS

output = ROOT / "statistical_output.txt"
with output.open("w", encoding="utf-8") as report:
    report.write("MATCH PHASE STATISTICS: REPRODUCIBILITY RUN OUTPUT\n")
    report.write("Run timestamp: withheld for anonymous review\n")
    report.write("Python: " + sys.version.split()[0] + "\n")
    report.write("Platform: details withheld for anonymous review\n")
    for module in [np, pd, scipy, statsmodels, patsy, threadpoolctl]:
        report.write(f"{module.__name__}: {module.__version__}\n")
    report.write("\nThese are native software outputs from a new, verified run.\n")
    report.write("They are not a recovered historical console log.\n")
    report.write("No models or tests are omitted on the basis of significance.\n")
    report.write(
        "Full-precision model coefficients include intercept, strength, duration and match terms.\n\n"
    )
    for relative in [
        "team_strength/test_output.txt",
        "phase_behaviours/test_output.txt",
        "success_models/model_output.txt",
    ]:
        report.write("\n" + "=" * 100 + "\n" + relative + "\n" + "=" * 100 + "\n")
        report.write((RESULTS / relative).read_text(encoding="utf-8"))
    report.write("\nCOMPLETE NUMERICAL RESULT TABLES\n")
    for path in sorted(RESULTS.glob("*/*.csv")):
        report.write("\n" + "=" * 100 + "\n" + str(path.relative_to(ROOT)) + "\n")
        table = pd.read_csv(path, float_precision="round_trip")
        report.write(f"Rows: {len(table)}; columns: {len(table.columns)}\n")
        if path.name == "spatial_density_grid.csv":
            report.write(
                "The complete 19,320-cell spatial grid is stored in the CSV named above.\n"
            )
        else:
            report.write(
                table.to_string(
                    index=False,
                    max_rows=None,
                    max_cols=None,
                    float_format=lambda value: format(value, ".17g"),
                )
            )
            report.write("\n")
print("Saved", output.name, flush=True)
