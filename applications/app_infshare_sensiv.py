import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from dc_workload_pipeline.workload_simulation import run_simulation  
from dc_workload_pipeline.config import OUTPUT_DIR, COOLING_COP, CITY

"""
This workbook analyzes AI data center workload sensitivity to inference shares (60–90%).
It evaluates how varying the inference-to-training ratio impacts daily power profiles, 
load-to-baseline deviations, and overall grid stability over a one-year period.
"""


# --- Parameter Set Sample Configuration ---
PARAMETER_SETS = [
    ("60%", (0.7, 1.0, 12, 20, 3.5, 3.5, 0.15, 0.85, 0.6, 1e8)),
    ("70%", (0.7, 1.0, 12, 20, 3.5, 3.5, 0.15, 0.85, 0.7, 1e8)),
    ("80%", (0.7, 1.0, 12, 20, 3.5, 3.5, 0.15, 0.85, 0.8, 1e8)),
    ("90%", (0.7, 1.0, 12, 20, 3.5, 3.5, 0.15, 0.85, 0.9, 1e8)),
]
STEPS_PER_DAY = 96

def run_analysis_pipeline():
    """Executes simulations and processes variability metrics."""
    all_results = {}
    variability_data = []

    for name, params in PARAMETER_SETS:
        # 1. Run Simulation
        results, _ = run_simulation(*params, COOLING_COP, CITY, print_results=False)
        all_results[name] = results

        # 2. Process Daily Variability (Vectorized)
        it_power = results["it_power"]
        it_baseline = results["it_baseline"]
        
        num_days = len(it_power) // STEPS_PER_DAY
        it_daily = it_power[:num_days * STEPS_PER_DAY].reshape(num_days, STEPS_PER_DAY)
        base_daily = it_baseline[:num_days * STEPS_PER_DAY].reshape(num_days, STEPS_PER_DAY)

        # Calculate metrics per day
        daily_min = np.min(it_daily, axis=1)
        daily_max = np.max(it_daily, axis=1)
        ratios = (daily_max - daily_min) / daily_min
        deviations = np.mean((it_daily - base_daily) / base_daily, axis=1)

        # Store for plotting
        for r, d in zip(ratios, deviations):
            variability_data.append({"Inference Share (%)": name, "Max-Min Ratio": r, "Deviation (%)": d})

    return all_results, pd.DataFrame(variability_data)

def plot_power_profiles(results_dict):
    """Generates stacked time-series for a 24-hour sample."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, sharey=True)
    
    for i, (name, res) in enumerate(results_dict.items()):
        t = res["t_hours"]
        # Limit to 48 hours for visual clarity if needed, or keep full
        axes[i].plot(t, res["inf_power"], label="Inference", color="crimson", lw=0.8)
        axes[i].plot(t, res["tr_power"], label="Training", color="deepskyblue", lw=0.8)
        axes[i].plot(t, res["cool_power"], label="Cooling", color="purple", lw=0.8)
        axes[i].plot(t, res["total_power"], label="Total", color="black", lw=1.5, ls='-')
        
        axes[i].set_title(f"Inference Share: {name}", fontweight="bold")
        axes[i].set_ylabel("Power (W)")
        axes[i].grid(True, alpha=0.3)
        if i == 0: axes[0].legend(loc="upper right", ncol=4)

    axes[-1].set_xlabel("Time (hours)")
    plt.tight_layout()
    plt.show()

def plot_variability_stats(df):
    """Plots statistical distributions of load volatility."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.violinplot(data=df, x="Inference Share (%)", y="Max-Min Ratio", ax=axes[0], palette="Blues")
    axes[0].set_title("Daily Max-to-Min Ratio", fontweight="bold")
    
    sns.violinplot(data=df, x="Inference Share (%)", y="Deviation (%)", ax=axes[1], palette="Reds")
    axes[1].set_title("Mean Deviation from Baseline", fontweight="bold")

    for ax in axes:
        ax.grid(axis='y', alpha=0.4)
        ax.set_xlabel("Inference Share (%)", fontweight="bold")

        ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    print("Starting sensitivity analysis...")
    sim_data, stats_df = run_analysis_pipeline()
    
    plot_power_profiles(sim_data)
    plot_variability_stats(stats_df)