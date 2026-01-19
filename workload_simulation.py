"""
Core Simulation Module: AI Data Center Flexible Load Simulation.

This module executes a high-fidelity simulation of data center power dynamics:
1. Temporal Grid: 15-minute intervals over a 365-day period.
2. Baselines: Diurnal Gaussian profiles for Inference utilization.
3. Power Scaling: Translates utilization to Watts using BioHPC regression.
4. Stochasticity: Ornstein-Uhlenbeck noise for high-frequency fluctuations.
5. Thermodynamics: Liquid and Free cooling power based on external temperatures.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

from dc_workload_pipeline.config import (
    DATA_DIR, OUTPUT_DIR, START_DATE, END_DATE, DELTA_T_MINUTES, IT_CAPACITY, POW_SERVER,
    AMP1, AMP2, MU1, MU2, SIGMA1, SIGMA2, BASELINE_FLOOR, BASELINE_CAP, INF_SHARE,
    BIOHPC_FILE, CITY, HOURS_PER_DAY, INTERVALS_PER_HOUR, COOLING_REF_TEMP,
    OU_THETA_INF, OU_SIGMA_INF, OU_THETA_TR, OU_SIGMA_TR,
    COOLING_H_S_A_S, COOLING_A, COOLING_U_POS, COOLING_U_NEG, COOLING_COP
)
from dc_workload_pipeline.biohpc_regression import compute_biohpc_parameters

# -------------------------------------------------
# TIME GRID INITIALIZATION
# -------------------------------------------------

timestamps = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq=f'{DELTA_T_MINUTES}T'
)

nb_intervals = len(timestamps)
nb_days = int(nb_intervals * DELTA_T_MINUTES / 60 / 24)
t_hours = np.arange(0, nb_days * 24, DELTA_T_MINUTES / 60)

# -------------------------------------------------
# CORE COMPUTATION FUNCTIONS
# -------------------------------------------------

def normalize(x, baseline_cap, baseline_floor):
    """Linear scaling of array x to target range [baseline_floor, baseline_cap]."""
    x_min, x_max = np.min(x), np.max(x)
    return ((x - x_min) / (x_max - x_min)) * (baseline_cap - baseline_floor) + baseline_floor


def generate_baselines(amp1, amp2, mu1, mu2, sigma1, sigma2, baseline_floor, baseline_cap, inf_share):
    """Generates the scheduled diurnal patterns for Inference utilization."""
    baseline_cap = np.minimum(1, np.random.normal(baseline_cap, 0.03))
    baseline = np.ones(len(t_hours)) * baseline_floor

    for k in range(nb_days):
        day_timestamp = timestamps[INTERVALS_PER_HOUR * HOURS_PER_DAY * k]
        is_weekend = day_timestamp.weekday() >= 5

        A1 = np.random.normal(amp1, 0.03)
        A2 = np.random.normal(amp2, 0.03)
        
        if is_weekend:
            A1 *= 0.5
            A2 *= 0.5

        # Double-peak Gaussian profile
        g1 = A1 * np.exp(-0.5 * ((t_hours - mu1 - k*24) / sigma1) ** 2)
        g2 = A2 * np.exp(-0.5 * ((t_hours - mu2 - k*24) / sigma2) ** 2)

        g1 -= np.min(g1)
        g2 -= np.min(g2)
        baseline += g1 + g2

    inf_baseline = normalize(baseline, 1, baseline_floor)
    return inf_baseline


def compute_power(utilization, slope, intercept, IT_capacity, biohpc_p_max):
    """Converts utilization percentage to power draw using regression coefficients."""
    return (slope * utilization * 100 + intercept) * IT_capacity / biohpc_p_max


def generate_ou_process(mean, theta, sigma, dt, n_steps):
    """Generates Ornstein-Uhlenbeck stochastic noise."""
    ou = np.zeros(n_steps)
    ou[0] = mean
    for t in range(1, n_steps):
        dW = np.random.normal(0, 1)
        ou[t] = ou[t-1] + theta * (mean - ou[t-1]) * dt + np.sqrt(dt) * sigma * dW
    return ou


def generate_power(amp1, amp2, mu1, mu2, sigma1, sigma2,
                   baseline_floor, baseline_cap, inf_share,
                   IT_capacity, slope, intercept, biohpc_p_max):
    """Builds full power profiles including baselines, training gaps, and noise."""
    
    # 1. Inference Baseline
    inf_baseline = generate_baselines(amp1, amp2, mu1, mu2, sigma1, sigma2, baseline_floor, baseline_cap, inf_share)
    inf_power_baseline = compute_power(inf_baseline, slope, intercept, IT_capacity, biohpc_p_max)

    # 2. Training Gap Filling
    raw_tr = IT_capacity * np.ones(len(inf_power_baseline)) - inf_power_baseline
    tr_power_baseline = raw_tr * ((1 - inf_share) / inf_share * (inf_power_baseline.sum() / raw_tr.sum()))

    # 3. Stochastic Noise Generation
    dt = DELTA_T_MINUTES / 60
    inf_noise = generate_ou_process(0, OU_THETA_INF, OU_SIGMA_INF * IT_capacity, dt, len(t_hours))
    tr_noise = generate_ou_process(0, OU_THETA_TR, OU_SIGMA_TR * IT_capacity, dt, len(t_hours))

    # Weighting noise by utilization intensity
    inf_weight = normalize(inf_power_baseline, 1, 0.5)
    tr_weight = normalize(tr_power_baseline, 1, 0.5)

    inf_power = inf_power_baseline + inf_noise * inf_weight
    tr_power = tr_power_baseline + tr_noise * tr_weight

    # 4. Global Hard-Cap Safety Check
    total_power = inf_power + tr_power
    if np.any(total_power > IT_capacity):
        scaling = IT_capacity / np.max(total_power)
        inf_power *= scaling
        tr_power *= scaling
        inf_power_baseline *= scaling
        tr_power_baseline *= scaling

    return inf_power, inf_power_baseline, tr_power, tr_power_baseline


def compute_cooling_power(it_power, T_ext, slope_pow_temp, intercept_pow_temp, biohpc_p_max,
                          IT_capacity, cop, pow_server=POW_SERVER, h_s_A_s=COOLING_H_S_A_S, 
                          A=COOLING_A, U_pos=COOLING_U_POS, U_neg=COOLING_U_NEG, T_ref_val=COOLING_REF_TEMP):
    """Computes total cooling load partitioned into liquid and free cooling."""
    
    # Internal Server Temperature estimation
    T_ser = slope_pow_temp * normalize(it_power, biohpc_p_max, 0) + intercept_pow_temp
    T_ref = np.ones(len(it_power)) * T_ref_val
    
    # Temperature-dependent thermal gain coefficient
    U = np.where(T_ext - T_ref >= 0, U_pos, U_neg)
    nb_servers = IT_capacity / pow_server
    
    # Power components (Watts)
    cool_power = nb_servers * (h_s_A_s * (T_ser - T_ref) + U * A * (T_ext - T_ref)) / cop
    liq = nb_servers * (h_s_A_s * (T_ser - T_ref)) / cop
    free = nb_servers * (U * A * (T_ext - T_ref)) / cop

    return cool_power, liq, free, T_ser, T_ref


# -------------------------------------------------
# EXECUTION INTERFACE
# -------------------------------------------------

def run_simulation(amp1, amp2, mu1, mu2, sigma1, sigma2,
                   baseline_floor, baseline_cap, inf_share,
                   IT_capacity, cop, city, print_results):
    """Main entry point to execute the simulation pipeline and generate metrics."""

    # 1. Acquire Hardware Regression Parameters
    _, reg_pow, reg_temp, biohpc_p_max = compute_biohpc_parameters(BIOHPC_FILE)
    slope_pow_util = reg_pow.slope
    intercept_pow_util = reg_pow.intercept
    slope_temp_pow = reg_temp.slope
    intercept_temp_pow = reg_temp.intercept



    # 2. Acquire Weather Data
    weather_file = DATA_DIR / f"nsrdb_{city}_2024.csv"
    weather_df = pd.read_csv(weather_file, header=2)
    T_ext = np.repeat(np.array(weather_df['Temperature']), 2)

    # 3. Generate IT Workloads
    inf_power, inf_base, tr_power, tr_base = generate_power(
        amp1, amp2, mu1, mu2, sigma1, sigma2,
        baseline_floor, baseline_cap, inf_share, IT_capacity,
        slope_pow_util, intercept_pow_util, biohpc_p_max
    )

    it_power = inf_power + tr_power
    it_baseline = inf_base + tr_base

    # 4. Generate Cooling Load
    cool, liq, free, T_ser, T_ref = compute_cooling_power(
        it_power, T_ext, slope_temp_pow, intercept_temp_pow, biohpc_p_max, IT_capacity, cop
    )

    total_power = it_power + cool

    # 5. Metrics & Summary Table
    inference_share = np.sum(inf_power) / np.sum(it_power)
    training_share = np.sum(tr_power) / np.sum(it_power)
    total_utilization = it_power / IT_capacity
    
    # Utilization statistics
    avg_util = np.mean(total_utilization)
    max_util = np.max(total_utilization)
    min_util = np.min(total_utilization)
    std_util = np.std(total_utilization)
    peak_to_avg_ratio = max_util / avg_util if avg_util > 0 else np.nan
    
    # Efficiency and Error Analysis
    pue = np.sum(total_power) / np.sum(it_power)
    avg_diff_total_perc = np.sum(np.abs(it_power - it_baseline)) / np.sum(it_baseline) * 100
    avg_diff_inf_perc = np.sum(np.abs(inf_power - inf_base)) / np.sum(it_baseline) * 100
    avg_diff_tr_perc = np.sum(np.abs(tr_power - tr_base)) / np.sum(it_baseline) * 100

    summary_df = pd.DataFrame({
        "Metric": [
            "Inference Power Share",
            "Training Power Share",
            "Average Total Utilization",
            "Maximum Utilization",
            "Minimum Utilization",
            "Std Dev Utilization",
            "Peak-to-Average Ratio",
            "Power Usage Effectiveness (PUE)",
            "Total Abs. Difference to IT Baseline (%)",
            "Inf. Abs. Difference to IT Baseline (%)",
            "Tra. Abs. Difference to IT Baseline (%)"
        ],
        "Value": [
            inference_share, training_share, avg_util, max_util, min_util,
            std_util, peak_to_avg_ratio, pue,            
            avg_diff_total_perc, avg_diff_inf_perc, avg_diff_tr_perc
        ]
    })

    if print_results:
        print("\n=== Simulation Summary Table ===")
        print(summary_df.to_string(index=False))

    return {
        "t_hours": t_hours,
        "inf_power": inf_power,
        "tr_power": tr_power,
        "inf_baseline": inf_base,
        "tr_baseline": tr_base,
        "it_power": it_power,
        "it_baseline": it_baseline,
        "cool_power": cool,
        "liq_cooling": liq,
        "free_cooling": free,
        "total_power": total_power,
        "T_ext": T_ext
    }, summary_df

if __name__ == "__main__":
    results, summary_df = run_simulation(
        AMP1, AMP2, MU1, MU2, SIGMA1, SIGMA2, 
        BASELINE_FLOOR, BASELINE_CAP, INF_SHARE, 
        IT_CAPACITY, COOLING_COP, CITY, True
    )
    print("\nSimulation completed successfully.")