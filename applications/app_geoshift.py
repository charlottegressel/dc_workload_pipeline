import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict

from dc_workload_pipeline.workload_simulation import run_simulation  
from dc_workload_pipeline.config import COOLING_COP, CITY


"""
This workbook evaluates how geographic diversification reduces grid pressure by comparing 6 independent NYC data centers 
against 6 data centers spread across different timezones. It measures min-to-max ratio and peak demand reduction across 
multiple simulation runs.
"""


# --- Configuration ---
PARAMETER_SETS = [
    ("New York",       (0.7, 1.0, 12, 20, 3.5, 3.5, 0.15, 0.85, 0.8, 1e8)),
    ("Chicago",        (0.7, 1.0, 13, 21, 3.5, 3.5, 0.15, 0.85, 0.8, 1e8)),
    ("Salt Lake City", (0.7, 1.0, 14, 22, 3.5, 3.5, 0.15, 0.85, 0.8, 1e8)),
    ("Los Angeles",    (0.7, 1.0, 15, 23, 3.5, 3.5, 0.15, 0.85, 0.8, 1e8)),
    ("Anchorage",      (0.7, 1.0, 16, 0,  3.5, 3.5, 0.15, 0.85, 0.8, 1e8)),
    ("Honolulu",       (0.7, 1.0, 17, 1,  3.5, 3.5, 0.15, 0.85, 0.8, 1e8)),
]

N_RUNS = 10

def get_aggregated_power(params_list: List[tuple]) -> np.ndarray:
    """
    Executes independent simulations for a list of parameter sets and returns 
    the sum of their total power profiles.
    """
    total_power = None
    for params in params_list:
        results, _ = run_simulation(*params, COOLING_COP, CITY, print_results=False)
        profile = results["total_power"]
        if total_power is None:
            total_power = profile.copy()
        else:
            total_power += profile
    return total_power

def collect_metrics(num_runs: int) -> pd.DataFrame:
    """
    Iterates through multiple simulation runs to build a statistical dataset
    comparing local vs. diversified DC clusters.
    """
    data_points = []
    nyc_params = PARAMETER_SETS[0][1]

    for run_id in range(num_runs):
        # Scenario 1: 6 Independent NYC simulations (Local Stochasticity)
        nyc_6_params = [nyc_params] * 6
        nyc_total = get_aggregated_power(nyc_6_params)

        # Scenario 2: 6 Different Timezones (Geographic Diversification)
        diversified_params = [p[1] for p in PARAMETER_SETS]
        diversified_total = get_aggregated_power(diversified_params)

        # Log metrics for both scenarios
        for label, power_arr in [("NYC Only (6 DCs)", nyc_total), 
                                 ("All Cities (6 DCs)", diversified_total)]:
            data_points.append({
                "Run": run_id,
                "Scenario": label,
                "Max/Min Ratio": power_arr.max() / power_arr.min(),
                "Max Demand (W)": power_arr.max()
            })

    return pd.DataFrame(data_points)

def plot_results(df: pd.DataFrame):
    """Generates distribution plots to visualize the diversification benefits."""
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Metrics to visualize
    plot_configs = [
        ("Max/Min Ratio", "Load Stability (Max/Min)"),
        ("Max Demand (W)", "Peak Aggregated Power")
    ]

    for i, (col, title) in enumerate(plot_configs):
        sns.violinplot(data=df, x="Scenario", y=col, ax=axes[i], 
                       inner="point", palette="Set2", hue="Scenario", legend=False)
        axes[i].set_title(title, fontweight="bold")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), fontweight="bold")
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(f"Running {N_RUNS} iterations...")
    results = collect_metrics(N_RUNS)

    # Quick terminal summary
    print("\n" + "-"*20 + " RESULTS SUMMARY " + "-"*20)
    summary = results.groupby("Scenario")[["Max/Min Ratio", "Max Demand (W)"]].mean()
    print(summary)

    # --- Peak demand reduction ---
    nyc_peak = summary.loc["NYC Only (6 DCs)", "Max Demand (W)"]
    geo_peak = summary.loc["All Cities (6 DCs)", "Max Demand (W)"]

    abs_reduction = nyc_peak - geo_peak
    pct_reduction = abs_reduction / nyc_peak * 100

    print("\n" + "-"*20 + " PEAK DEMAND REDUCTION " + "-"*20)
    print(f"NYC-only peak demand:        {nyc_peak:,.0f} W")
    print(f"Diversified peak demand:    {geo_peak:,.0f} W")
    print(f"Absolute reduction:         {abs_reduction:,.0f} W")
    print(f"Percentage reduction:       {pct_reduction:.2f} %")

    plot_results(results)
