"""
This workbook evaluates and compares five distinct data center power profiles.
It simulates various IT workload scenarios (varying peak amplitudes, shifts, 
variability, and capacity) and exports a comprehensive Excel summary containing 
both simulation parameters and key performance metrics, alongside a visual comparison.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import openpyxl

from dc_workload_pipeline.workload_simulation import run_simulation  
from dc_workload_pipeline.config import OUTPUT_DIR, COOLING_COP, CITY

# --- Configuration & Parameter Sets ---
PARAMETER_SETS = [
    ("Profile 1 (Baseline)", 
     (0.7, 1.0, 12, 20, 3.5, 3.5, 0.15, 0.85, 0.7, 100_000_000)),
    ("Profile 2 (Higher Peak Amplitudes)", 
     (1.0, 1.3, 12, 20, 3.5, 3.5, 0.15, 0.85, 0.7, 100_000_000)),
    ("Profile 3 (Shifted Peaks)", 
     (0.7, 1.0, 10, 18, 3.5, 3.5, 0.15, 0.85, 0.7, 100_000_000)),
    ("Profile 4 (More Short-term Variability)", 
     (0.7, 1.0, 12, 20, 2.0, 2.0, 0.15, 0.85, 0.7, 100_000_000)),
    ("Profile 5 (Higher Capacity & Inference share)", 
     (0.7, 1.0, 12, 20, 3.5, 3.5, 0.15, 0.90, 0.85, 130_000_000)),
]

EXCEL_PATH = os.path.join(OUTPUT_DIR, "5profiles_summary.xlsx")
PLOT_PATH = os.path.join(OUTPUT_DIR, "multi_profile_comparison.png")

def run_multi_profile_analysis():
    """Executes simulation for all profiles, saves to Excel, and generates plots."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    writer = pd.ExcelWriter(EXCEL_PATH, engine="openpyxl")
    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True, sharey=True)

    for i, (name, params) in enumerate(PARAMETER_SETS):
        # 1. Run Simulation
        results, summary_df = run_simulation(*params, COOLING_COP, CITY, print_results=False)
        
        # 2. Data Preparation
        t = results["t_hours"]
        it_baseline = results["inf_baseline"] + results["tr_baseline"]
        
        # 3. Excel Export (Parameters + Summary)
        param_labels = [
            "amp1", "amp2", "mu1", "mu2", "sigma1", "sigma2",
            "baseline_floor", "baseline_cap", "inf_share", "it_capacity", "cop"
        ]
        param_values = list(params) + [COOLING_COP]
        
        param_df = pd.DataFrame({"Parameter": param_labels, "Value": param_values})
        sheet_name = f"Profile_{i+1}"
        
        param_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
        summary_df.iloc[:, :2].to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(param_df) + 3)

       # 4. Visualization (Slicing for the first 7 days: 96 intervals * 7)
        view_slice = slice(0, 96 * 7)
        
        t_slice = t[view_slice]
        inf_slice = results["inf_power"][view_slice]
        tr_slice = results["tr_power"][view_slice]
        cool_slice = results["cool_power"][view_slice]
        base_slice = it_baseline[view_slice]
        total_slice = results["total_power"][view_slice]

        # Plotting sliced data
        axes[i].plot(t_slice, inf_slice, label="Inference", color="crimson", lw=1.0, drawstyle="steps-post")
        axes[i].plot(t_slice, tr_slice, label="Training", color="deepskyblue", lw=1.0, drawstyle="steps-post")
        axes[i].plot(t_slice, cool_slice, label="Cooling", color="purple", lw=1.0, drawstyle="steps-post")
        axes[i].plot(t_slice, base_slice, label="IT Baseline", color="blue", lw=1.2, ls='--', drawstyle="steps-post")
        axes[i].plot(t_slice, total_slice, label="Total Power", color="black", lw=1.8, drawstyle="steps-post")

        # Styling
        axes[i].set_title(name, fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Power (W)", fontweight="bold")
        axes[i].grid(True, alpha=0.3)
        
        # Ensure the x-axis limits are tight to the 5-day window
        axes[i].set_xlim(t_slice.min(), t_slice.max())
        
        # Bold Ticks
        for label in axes[i].get_yticklabels() + axes[i].get_xticklabels():
            label.set_fontweight('bold')

        if i == 0:
            axes[i].legend(fontsize=10, loc='upper right', ncol=5)

    axes[-1].set_xlabel("Time (hours)", fontweight="bold")
    
    # Finalize Excel
    writer.close()
    print(f"Summary data saved to: {EXCEL_PATH}")

    # Save and Show Plot
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {PLOT_PATH}")
    plt.show()

if __name__ == "__main__":
    run_multi_profile_analysis()