import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from dc_workload_pipeline.workload_simulation import run_simulation 
from dc_workload_pipeline.config import (
    AMP1, AMP2, MU1, MU2, SIGMA1, SIGMA2,
    BASELINE_FLOOR, BASELINE_CAP, INF_SHARE, IT_CAPACITY
)

"""
This workbook evaluates Data Center Power Usage Effectiveness (PUE) sensitivity to Cooling COP.
It simulates various U.S. cities to capture ambient temperature impacts and highlights 
performance benchmarks for traditional Air-Cooling (CRAC) vs. advanced Liquid-Cooling.
"""


# --- Configuration & Benchmarks ---
COP_VALUES = np.linspace(3, 20, 40)
CITIES = ["newyork", "chicago", "memphis", "saltlakecity", "seattle", "losangeles"]
COP_CRAC = 5.366
COP_LIQ = 11.7

def run_pue_sensitivity():
    """Iterates through cities and COP values to extract PUE metrics."""
    results_list = []

    for city in CITIES:
        print(f"Simulating PUE trends for: {city}...")
        for cop in COP_VALUES:
            _, summary_df = run_simulation(
                AMP1, AMP2, MU1, MU2, SIGMA1, SIGMA2,
                BASELINE_FLOOR, BASELINE_CAP, INF_SHARE,
                IT_CAPACITY,
                cop=cop,
                city=city,
                print_results=False
            )

            # Extract PUE from the summary dataframe
            pue = summary_df.loc[
                summary_df["Metric"] == "Power Usage Effectiveness (PUE)", "Value"
            ].values[0]

            results_list.append({
                "City": city.capitalize(),
                "COP": cop,
                "PUE": pue
            })

    return pd.DataFrame(results_list)

def plot_pue_trends(df):
    """Generates a professional visualization of PUE vs COP."""
    plt.figure(figsize=(11, 7))
    
    # 1. Plot Individual City Trends
    for city in df["City"].unique():
        city_data = df[df["City"] == city]
        plt.plot(city_data["COP"], city_data["PUE"], lw=1.2, alpha=0.4, label=city)

    # 2. Calculate and Plot Average Trend
    avg_curve = df.groupby("COP")["PUE"].mean()
    plt.plot(avg_curve.index, avg_curve.values, color="black", lw=2.5, label="U.S. Average")

    # 3. Handle Benchmarks (Interp for accuracy)
    avg_crac = np.interp(COP_CRAC, avg_curve.index, avg_curve.values)
    avg_liq = np.interp(COP_LIQ, avg_curve.index, avg_curve.values)

    # Annotations
    plt.axvline(COP_CRAC, ls="--", lw=1, color="gray", alpha=0.7)
    plt.axvline(COP_LIQ, ls="--", lw=1, color="gray", alpha=0.7)
    plt.scatter([COP_CRAC, COP_LIQ], [avg_crac, avg_liq], color="black", zorder=5)

    plt.text(COP_CRAC, avg_crac + 0.005, f" Air: {avg_crac:.3f}", fontweight="bold", ha="right")
    plt.text(COP_LIQ, avg_liq + 0.005, f" Liquid: {avg_liq:.3f}", fontweight="bold", ha="left")

    # --- Bold Styling & Formatting ---
    plt.title("Data Center PUE Sensitivity to Cooling Efficiency (COP)", fontweight="bold", fontsize=14)
    plt.xlabel("Coefficient of Performance (COP)", fontweight="bold")
    plt.ylabel("Power Usage Effectiveness (PUE)", fontweight="bold")
    
    ax = plt.gca()
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", frameon=True)
    

if __name__ == "__main__":
    pue_df = run_pue_sensitivity()
    plot_pue_trends(pue_df)