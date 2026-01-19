"""
This module performs linear regression analysis on BioHPC server data.
It extracts relationship coefficients between GPU utilization, power draw, 
and operating temperatures to parameterize the broader workload simulation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from dc_workload_pipeline.config import DATA_DIR, OUTPUT_DIR

# --- Configuration ---
FILE_NAME = "cbsugpu08_usage_stats.txt"
SAVE_PLOTS = True

def compute_biohpc_parameters(file_name=FILE_NAME):
    """
    Cleans BioHPC telemetry data and computes linear regression parameters 
    for Power-vs-Utilization and Temperature-vs-Power.
    """
    path = DATA_DIR / file_name
    
    # Load and clean data
    df = pd.read_csv(path, sep='\t')
    df['powerave'] = pd.to_numeric(df['powerave'], errors='coerce')
    df['utilave'] = pd.to_numeric(df['utilave'], errors='coerce')
    df['tempave'] = pd.to_numeric(df['tempave'], errors='coerce')

    # Filter out inactive states and NaNs
    mask = (df['powerave'] > 0) & (df['utilave'] > 0)
    df_clean = df[mask].dropna(subset=['powerave', 'utilave', 'tempave'])

    # 1. Power vs Utilization Regression
    res_util = linregress(df_clean['utilave'], df_clean['powerave'])
    
    # 2. Temperature vs Power Regression
    res_temp = linregress(df_clean['powerave'], df_clean['tempave'])
    
    # Metrics
    p_max = df_clean['powerave'].max()
    
    return df_clean, res_util, res_temp, p_max

def plot_biohpc_analysis(df, res_util, res_temp):
    """Generates professional regression and distribution plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Power vs Utilization
    axes[0].scatter(df['utilave'], df['powerave'], color='royalblue', alpha=0.2, s=10)
    x_range = np.array([df['utilave'].min(), df['utilave'].max()])
    axes[0].plot(x_range, res_util.intercept + res_util.slope * x_range, color='red', lw=2)
    axes[0].set_title(f"Power vs Utilization ($R^2$: {res_util.rvalue**2:.3f})", fontweight='bold')
    axes[0].set_xlabel("Utilization (%)", fontweight='bold')
    axes[0].set_ylabel("Power (W)", fontweight='bold')

    # Plot 2: Temperature vs Power
    axes[1].scatter(df['powerave'], df['tempave'], color='seagreen', alpha=0.2, s=10)
    x_range_temp = np.array([df['powerave'].min(), df['powerave'].max()])
    axes[1].plot(x_range_temp, res_temp.intercept + res_temp.slope * x_range_temp, color='red', lw=2)
    axes[1].set_title(f"Temp vs Power ($R^2$: {res_temp.rvalue**2:.3f})", fontweight='bold')
    axes[1].set_xlabel("Power (W)", fontweight='bold')
    axes[1].set_ylabel("Temperature (°C)", fontweight='bold')

    # Plot 3: Power Distribution (Boxplot)
    df['util_bin'] = pd.cut(df['utilave'], bins=np.arange(0, 110, 10))
    grouped = [group['powerave'].values for _, group in df.groupby('util_bin', observed=True)]
    labels = [str(int(b.left)) for b in df['util_bin'].cat.categories]
    
    axes[2].boxplot(grouped, labels=labels, patch_artist=True, 
                    boxprops=dict(facecolor='whitesmoke'), medianprops=dict(color='red'))
    axes[2].set_title("Power Distribution by Util Bin", fontweight='bold')
    axes[2].set_xlabel("Utilization Decile (%)", fontweight='bold')
    axes[2].set_ylabel("Power (W)", fontweight='bold')

    # Global Styling
    for ax in axes:
        ax.grid(True, alpha=0.3)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(OUTPUT_DIR / "biohpc_characterization.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Execute Analysis
    df_clean, reg_pow, reg_temp, p_max = compute_biohpc_parameters()

    # Log Results
    print(f"{'='*30}\nBIOHPC PARAMETERS\n{'='*30}")
    print(f"Power-Util Slope:     {reg_pow.slope:.4f}")
    print(f"Power-Util Intercept: {reg_pow.intercept:.4f}")
    print(f"Peak Observed Power:  {p_max:.2f} W")
    print(f"Temp-Power Slope:     {reg_temp.slope:.4f}")
    print(f"{'='*30}")

    # Generate Visualization
    plot_biohpc_analysis(df_clean, reg_pow, reg_temp)