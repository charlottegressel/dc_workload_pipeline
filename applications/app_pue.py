"""
This workbook analyzes seasonal Power Usage Effectiveness (PUE) variations for a 
New York data center. It categorizes PUE data into Summer, Winter, and Mid-season, 
visualizing hourly distributions via boxplots to highlight the impact of 
ambient temperature on cooling efficiency throughout the year.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dc_workload_pipeline.config import (
    AMP1, AMP2, MU1, MU2, SIGMA1, SIGMA2,
    BASELINE_FLOOR, BASELINE_CAP, INF_SHARE,
    IT_CAPACITY, COOLING_COP
)
from dc_workload_pipeline.workload_simulation import run_simulation  

# --- Configuration & Setup ---
CITY = "newyork"
OUTPUT_DIR = "outputs"
INTERVALS_PER_DAY = 96  # 15-min intervals
Y_LIMITS = (1.10, 1.24)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_seasonal_data():
    """Runs simulation and segments PUE by season."""
    data, _ = run_simulation(
        AMP1, AMP2, MU1, MU2, SIGMA1, SIGMA2,
        BASELINE_FLOOR, BASELINE_CAP, INF_SHARE, IT_CAPACITY,
        COOLING_COP, CITY, print_results=False
    )

    # Calculate PUE and structure DataFrame
    df = pd.DataFrame({
        "PUE": data["total_power"] / data["it_power"],
        "hour": (np.arange(len(data["it_power"])) % INTERVALS_PER_DAY) // 4,
        "day": np.arange(len(data["it_power"])) // INTERVALS_PER_DAY
    })

    # Define seasonal ranges
    summer_days = range(172, 264)
    winter_days = list(range(0, 79)) + list(range(355, 365))
    mid_days = list(range(79, 172)) + list(range(264, 355))

    def get_hourly_dist(day_list):
        season_df = df[df["day"].isin(day_list)]
        hourly_pue = season_df.groupby(["day", "hour"])["PUE"].mean().reset_index()
        return [hourly_pue[hourly_pue["hour"] == h]["PUE"].values for h in range(24)]

    return {
        "Summer": get_hourly_dist(summer_days),
        "Mid-season": get_hourly_dist(mid_days),
        "Winter": get_hourly_dist(winter_days)
    }

def plot_seasonal_pue(seasonal_results):
    """Generates a professional three-pane seasonal PUE comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Common Styling Helper
    def style_subplot(ax, title, data):
        # Calculate Mean
        flat_data = np.concatenate(data)
        season_mean = flat_data.mean()
        
        # Plot Boxplot
        ax.boxplot(data, patch_artist=True, 
                   boxprops=dict(facecolor='whitesmoke', color='black'),
                   medianprops=dict(color='red', linewidth=1.5))
        
        # Add Mean Reference Line
        ax.axhline(season_mean, linestyle='--', color='red', alpha=0.6, label="Mean")
        
        # Labels and Formatting
        ax.set_title(title, fontweight="bold", fontsize=14, pad=25)
        ax.set_xlabel("Time (hours)", fontweight="bold")
        ax.set_ylim(Y_LIMITS)
        
        # Annotate Mean PUE
        ax.text(0.5, 1.02, f"Mean PUE: {season_mean:.3f}", 
                transform=ax.transAxes, ha='center', fontsize=11, color='dimgray', fontweight='bold')

        # Bold Axis Ticks
        ax.set_xticks(np.arange(1, 25, 2))  # Boxplot x-axis starts at 1
        ax.set_xticklabels([str(i) for i in np.arange(0, 24, 2)])
        for label in ax.get_yticklabels() + ax.get_xticklabels():
            label.set_fontweight('bold')
        
        ax.grid(axis='y', alpha=0.3)

    # Draw Subplots
    style_subplot(axes[0], "Summer", seasonal_results["Summer"])
    style_subplot(axes[1], "Mid-season", seasonal_results["Mid-season"])
    style_subplot(axes[2], "Winter", seasonal_results["Winter"])
    
    axes[0].set_ylabel("Power Usage Effectiveness (PUE)", fontweight="bold")

    plt.tight_layout()
    
    # Save Output
    save_path = os.path.join(OUTPUT_DIR, "seasonal_pue_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Seasonal analysis saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    print(f"Starting Seasonal PUE Analysis for {CITY.upper()}...")
    seasonal_data = process_seasonal_data()
    plot_seasonal_pue(seasonal_data)