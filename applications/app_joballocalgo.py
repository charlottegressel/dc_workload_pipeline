"""
This workbook simulates training power adapting to an existing inference workload over two days. 
Training tasks are modeled as a non-linear queue depending on power, priority, and duration. 
A job allocation algorithm schedules tasks within the available inference and maximum capacity. 
It produces plots of the overall IT power profile and summary tables to analyze power usage, 
variability, and job completion metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dc_workload_pipeline.config import AMP1, AMP2, MU1, MU2, SIGMA1, SIGMA2
from dc_workload_pipeline.workload_simulation import (
    generate_power, compute_biohpc_parameters, timestamps,
    BIOHPC_FILE, BASELINE_FLOOR, BASELINE_CAP, INF_SHARE
)

# --- Configuration ---
OUTPUT_DIR = "outputs"
NUM_JOBS = 100
JOB_PROB_DIST = [0.55, 0.35, 0.10]  # Short, Medium, Long
STEPS_PER_DAY = 96
JOB_COLORS = ['skyblue', 'lightgreen', 'salmon']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_random_jobs(num_jobs, slope, intercept):
    """Generates a synthetic workload of training jobs with varying priorities."""
    job_types = np.random.choice(['short', 'medium', 'long'], size=num_jobs, p=JOB_PROB_DIST)
    job_list = []

    for i, jt in enumerate(job_types):
        if jt == 'short':
            dur, util, prio = np.random.randint(2, 8), np.random.uniform(10, 40), np.random.normal(2, 1)
        elif jt == 'medium':
            dur, util, prio = np.random.randint(8, 32), np.random.uniform(40, 80), np.random.normal(2, 1)
        else:
            dur, util, prio = np.random.randint(32, 96), np.random.uniform(80, 100), np.random.normal(3, 1.5)
        
        power = slope * util + intercept
        job_list.append({'job_id': f'J{i}', 'duration': dur, 'util': util, 'job_type': jt, 'priority': prio, 'power': power})
    
    return pd.DataFrame(job_list).sort_values(by=['priority', 'duration', 'power'], ascending=False)

def run_job_allocation():
    """Main simulation pipeline for inference and adaptive training allocation."""
    _, reg_pow, _, biohpc_p_max = compute_biohpc_parameters(BIOHPC_FILE)
    slope = reg_pow.slope
    intercept = reg_pow.intercept
    p_max = biohpc_p_max * 8  # 8x H100 System
    
    ts_subset = timestamps[:STEPS_PER_DAY * 2]
    T = len(ts_subset)
    inf_p, inf_base, _, _ = generate_power(
        AMP1, AMP2, MU1, MU2, SIGMA1, SIGMA2,
        BASELINE_FLOOR, BASELINE_CAP, INF_SHARE, p_max,
        slope, intercept, biohpc_p_max
    )

    df = pd.DataFrame({
        'timestamp': ts_subset,
        'Inference Power (W)': inf_p[:T],
        'Inference Baseline Power (W)': inf_base[:T]
    })

    jobs = generate_random_jobs(NUM_JOBS, slope, intercept)
    avail_p = (p_max - df['Inference Power (W)']).values
    df['Training Power (W)'] = 0.0
    completed_jobs = []

    for _, job in jobs.iterrows():
        dur, pwr = int(job['duration']), job['power']
        for t_start in range(T - dur):
            if np.all(avail_p[t_start:t_start + dur] >= pwr):
                avail_p[t_start:t_start + dur] -= pwr
                df.loc[t_start:t_start + dur - 1, 'Training Power (W)'] += pwr
                completed_jobs.append({**job.to_dict(), 'start': ts_subset[t_start]})
                break

    df = df.iloc[:STEPS_PER_DAY].copy()
    df['Training Baseline (W)'] = p_max - df['Inference Baseline Power (W)']
    df['Total Power (W)'] = df['Inference Power (W)'] + df['Training Power (W)']
    df['Variability (%)'] = 100 * (df['Training Power (W)'] - df['Training Baseline (W)']) / df['Training Baseline (W)']
    
    return df, pd.DataFrame(completed_jobs), p_max

def plot_allocation_profile(df, p_max):
    """Generates the dual-axis power and variability dashboard."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    axes[0].plot(df['timestamp'], df['Inference Power (W)'], label='Inference', color='crimson', drawstyle='steps-post')
    axes[0].plot(df['timestamp'], df['Training Baseline (W)'], label='Training Baseline', color='lightskyblue', ls='--', alpha=0.7)
    axes[0].plot(df['timestamp'], df['Training Power (W)'], label='Training (Scheduled)', color='deepskyblue', lw=2, drawstyle='steps-post')
    axes[0].plot(df['timestamp'], df['Total Power (W)'], label='Total IT Load', color='black', ls='-', lw=1.5)
    axes[0].axhline(y=p_max, color='orange', ls=':', label='Server Capacity')
    
    axes[0].set_title('8x H100 Server Power Allocation (24h Window)', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('Power (W)', fontweight='bold')
    
    axes[1].plot(df['timestamp'], df['Variability (%)'], label='Training vs Baseline', color='forestgreen', drawstyle='steps-post')
    axes[1].axhline(df['Variability (%)'].mean(), color='orange', ls=':', label='Mean Deviation')
    axes[1].set_ylabel('Deviation (%)', fontweight='bold')
    axes[1].set_xlabel('Time (Hours)', fontweight='bold')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', ncol=2)
        for label in ax.get_yticklabels() + ax.get_xticklabels():
            label.set_fontweight('bold')
            
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_allocation_profile.png"), dpi=300)
    plt.show()

def plot_job_statistics(completed_df):
    """Generates a statistical breakdown of completed training tasks."""
    job_types = ['short', 'medium', 'long']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # 1. Duration Histogram
    axes[0].hist(
        [completed_df[completed_df['job_type'] == jt]['duration'] * 15 for jt in job_types],
        bins=20, stacked=True, color=JOB_COLORS, label=job_types, edgecolor='black'
    )
    axes[0].set_title('Distribution of Completed Job Durations', fontweight='bold')
    axes[0].set_xlabel('Job Duration (minutes)', fontweight='bold')
    axes[0].set_ylabel('Number of Jobs', fontweight='bold')

    # 2. Power Histogram
    axes[1].hist(
        [completed_df[completed_df['job_type'] == jt]['power'] for jt in job_types],
        bins=20, stacked=True, color=JOB_COLORS, label=job_types, edgecolor='black'
    )
    axes[1].set_title('Distribution of Completed Job Power', fontweight='bold')
    axes[1].set_xlabel('Job Power (W)', fontweight='bold')
    axes[1].set_ylabel('Number of Jobs', fontweight='bold')

    # 3. Priority vs Duration
    for jt, color in zip(job_types, JOB_COLORS):
        subset = completed_df[completed_df['job_type'] == jt]
        axes[2].scatter(subset['duration'] * 15, subset['priority'], color=color, label=jt, alpha=0.7, s=50, edgecolors='black')
    axes[2].set_title('Job Priority vs Duration', fontweight='bold')
    axes[2].set_xlabel('Job Duration (minutes)', fontweight='bold')
    axes[2].set_ylabel('Priority', fontweight='bold')

    # 4. Priority vs Power
    for jt, color in zip(job_types, JOB_COLORS):
        subset = completed_df[completed_df['job_type'] == jt]
        axes[3].scatter(subset['power'], subset['priority'], color=color, label=jt, alpha=0.7, s=50, edgecolors='black')
    axes[3].set_title('Job Priority vs Power', fontweight='bold')
    axes[3].set_xlabel('Job Power (W)', fontweight='bold')
    axes[3].set_ylabel('Priority', fontweight='bold')

    # Styling and Save
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
        for label in ax.get_yticklabels() + ax.get_xticklabels():
            label.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_job_statistics.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    print("Starting Training Allocation Simulation...")
    df_results, df_jobs, max_capacity = run_job_allocation()
    
    print(f"Total Jobs Scheduled: {len(df_jobs)}/{NUM_JOBS} ({len(df_jobs)/NUM_JOBS:.1%})")
    
    # Generate Plots
    plot_allocation_profile(df_results, max_capacity)
    plot_job_statistics(df_jobs)