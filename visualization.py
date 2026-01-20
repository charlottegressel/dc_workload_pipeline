"""
Visualization Module: AI Data Center Workload Analysis.
Generates performance dashboards including power allocation profiles, 
stochastic noise analysis, and hourly error distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dc_workload_pipeline.config import (
    AMP1, AMP2, MU1, MU2, SIGMA1, SIGMA2, 
    BASELINE_FLOOR, BASELINE_CAP, INF_SHARE, 
    IT_CAPACITY, COOLING_COP, CITY, OUTPUT_DIR
)
from dc_workload_pipeline.workload_simulation import run_simulation

# --- 1. Data Acquisition ---
data, summary_df = run_simulation(
    AMP1, AMP2, MU1, MU2, SIGMA1, SIGMA2, 
    BASELINE_FLOOR, BASELINE_CAP, INF_SHARE, 
    IT_CAPACITY, COOLING_COP, CITY, print_results=False
)

# Constants for slicing (Days 5 to 12 for diurnal clarity)
START_SLICE, END_SLICE = 96*5, 96*12
t_window = data["t_hours"][:(END_SLICE - START_SLICE)]

def plot_power_breakdown(data):
    """Main dashboard showing power tiers and cooling components."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    t = data["t_hours"][START_SLICE:END_SLICE]

    # Subplot 0: Overview
    axes[0].plot(t, data["inf_power"][START_SLICE:END_SLICE], label='Inference', color="crimson", lw=0.8, drawstyle='steps-post')
    axes[0].plot(t, data["tr_power"][START_SLICE:END_SLICE], label='Training', color="deepskyblue", lw=0.8, drawstyle='steps-post')
    axes[0].plot(t, data["cool_power"][START_SLICE:END_SLICE], label='Cooling', color="purple", lw=0.8, drawstyle='steps-post')
    axes[0].plot(t, data["total_power"][START_SLICE:END_SLICE], label='Total Load', color="black", lw=1.5, drawstyle='steps-post')
    axes[0].set_title("Total Data Center Power Profile (7-Day Sample)", fontweight='bold')

    # Subplot 1: IT Workload vs Baselines
    axes[1].plot(t, data["inf_power"][START_SLICE:END_SLICE], color="crimson", alpha=0.6, lw=0.8, drawstyle='steps-post')
    axes[1].plot(t, data["inf_baseline"][START_SLICE:END_SLICE], label='Inf. Baseline', color="darkred", ls=':', lw=1.2)
    axes[1].plot(t, data["tr_power"][START_SLICE:END_SLICE], color="deepskyblue", alpha=0.6, lw=0.8, drawstyle='steps-post')
    axes[1].plot(t, data["tr_baseline"][START_SLICE:END_SLICE], label='Tr. Baseline', color="darkblue", ls=':', lw=1.2)
    axes[1].set_title("Workload Fluctuations vs. Scheduled Baselines", fontweight='bold')

    # Subplot 2: Cooling Breakdown (Stacked Area)
    it, liq, free = data["it_power"][START_SLICE:END_SLICE], data["liq_cooling"][START_SLICE:END_SLICE], data["free_cooling"][START_SLICE:END_SLICE]
    axes[2].plot(t, it, color='black', lw=1, label='IT Load')
    axes[2].fill_between(t, it, it + liq, color='blue', alpha=0.3, label='Liquid Cooling', step='post')
    axes[2].fill_between(t, it + liq, it + liq + free, color='cyan', alpha=0.3, label='Air/Free Cooling', step='post')
    axes[2].set_title("Cooling Component Stacking", fontweight='bold')

    for ax in axes:
        ax.set_ylabel('Power (W)', fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3)
        for label in ax.get_yticklabels() + ax.get_xticklabels():
            label.set_fontweight('bold')

    plt.xlabel('Time (hours)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "power_profile_dashboard.png", dpi=300)

def plot_stochastic_noise(data):
    """Analysis of Gaussian diurnal patterns and short-term noise."""
    inf = data["inf_power"][START_SLICE:END_SLICE]
    inf_b = data["inf_baseline"][START_SLICE:END_SLICE]
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_window, inf_b, label='Scheduled Baseline', color="orange", lw=2)
    plt.plot(t_window, inf, label='Actual Load (with noise)', color="royalblue", alpha=0.7, drawstyle="steps-post")
    plt.plot(t_window, inf - inf_b, label='Stochastic Residual', color="gray", alpha=0.5, lw=0.8)
    
    plt.title("Stochastic Workload Analysis (7-Day Sample)", fontweight='bold')
    plt.ylabel('Power (W)', fontweight='bold')
    plt.xlabel('Hours from Start', fontweight='bold')
    plt.legend(ncol=3, loc='upper center')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_hourly_error_dist(data):
    """Boxplots showing absolute error distributions by hour of day."""
    t = data["t_hours"]
    it, it_b = data["it_power"], data["it_baseline"]
    
    # Calculation
    it_diff = np.abs(it - it_b) / it_b
    hours = np.floor(t % 24).astype(int)
    
    hourly_data = [it_diff[hours == h] for h in range(24)]

    plt.figure(figsize=(12, 5))
    # Boxplot
    bp = plt.boxplot(hourly_data, positions=np.arange(24), patch_artist=True,
                     boxprops=dict(facecolor='whitesmoke', alpha=0.5))
    
    # Jittered points (subset for performance)
    for h in range(24):
        y = hourly_data[h]
        if len(y) > 100: y = np.random.choice(y, 100) # Subset to keep responsive
        x = np.random.normal(h, 0.1, size=len(y))
        plt.scatter(x, y, alpha=0.3, s=5, color='royalblue')

    plt.title("Total IT Workload: Absolute Error to Baseline by Hour", fontweight='bold')
    plt.ylabel("Abs. Error (%)", fontweight='bold')
    plt.xlabel("Hour of Day", fontweight='bold')
    plt.xticks(np.arange(0, 24, 2), fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Execute Plotting ---
if __name__ == "__main__":
    plot_power_breakdown(data)
    plot_stochastic_noise(data)
    plot_hourly_error_dist(data)