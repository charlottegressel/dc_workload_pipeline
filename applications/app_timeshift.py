import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

from dc_workload_pipeline.workload_simulation import run_simulation
from dc_workload_pipeline.config import IT_CAPACITY, COOLING_COP, CITY

"""
This workbook simulates load-shifting strategies across five data center power profiles, 
comparing optimization performance at 15-minute and hourly resolutions. 
It specifically evaluates differences in peak reduction efficacy under high-demand scenarios.
"""


class WorkloadOptimizer:
    """
    Optimizes multi-DC IT power profiles to reduce peaks and variability.
    Supports analysis at 15-minute and hourly resolutions for comparison.
    """

    def __init__(self, parameter_sets: List[Tuple[str, tuple]], peak_threshold: float = 1e8):
        self.parameter_sets = parameter_sets
        self.peak_threshold = peak_threshold
        self.timesteps_per_day = 96
        self.results = {}

    def run_simulations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Simulates IT power profiles based on provided parameter sets."""
        simulations = {}
        for name, params in self.parameter_sets:
            data, _ = run_simulation(*params, COOLING_COP, CITY, False)
            simulations[name] = {
                "it_power": data["it_power"], 
                "it_baseline": data["it_baseline"]
            }
        return simulations

    def _find_peak_day(self, all_powers: np.ndarray) -> int:
        """Locates the first day where power exceeds the peak threshold."""
        n_profiles, total_steps = all_powers.shape
        n_days = total_steps // self.timesteps_per_day
        
        for d in range(n_days):
            start = d * self.timesteps_per_day
            end = start + self.timesteps_per_day
            if np.any(all_powers[:, start:end] >= self.peak_threshold):
                return d
        raise ValueError(f"No day contains a peak of {self.peak_threshold}")

    @staticmethod
    def to_hourly(data: np.ndarray) -> np.ndarray:
        """Vectorized conversion of 15-min intervals to hourly means."""
        n_profiles, n_steps = data.shape
        return np.mean(data.reshape(n_profiles, -1, 4), axis=2)

    def optimize_load(self, data: np.ndarray, shift_limit: float, target_mean: float) -> np.ndarray:
        """
        Solves the convex optimization problem to minimize deviation from target mean.
        """
        n_profiles, n_steps = data.shape
        optimized = np.zeros_like(data)
        
        for t in range(n_steps):
            p = cp.Variable(n_profiles)
            actual_t = data[:, t]
            
            # Objective: Minimize absolute deviation from the global daily mean
            objective = cp.Minimize(cp.sum(cp.abs(p - target_mean)))
            
            constraints = [
                cp.sum(p) == np.sum(actual_t),
                p >= 0,
                p >= (1 - shift_limit) * actual_t,
                p <= (1 + shift_limit) * actual_t
            ]
            
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS) # Robust solver choice
            optimized[:, t] = p.value
            
        return optimized

    def analyze(self, shift_15min: float = 0.5, shift_hourly: float = 0.5):
        """Executes the full optimization and analysis pipeline."""
        sims = self.run_simulations()
        
        # Prepare Data
        all_powers = np.array([sims[name]["it_power"] for name, _ in self.parameter_sets])
        peak_day = self._find_peak_day(all_powers)
        
        start = peak_day * self.timesteps_per_day
        end = start + self.timesteps_per_day
        
        it_powers = all_powers[:, start:end]
        it_baselines = np.array([sims[name]["it_baseline"][start:end] for name, _ in self.parameter_sets])
        
        # Calculate Target
        global_mean = np.mean(it_powers)
        
        # Optimization
        opt_15 = self.optimize_load(it_powers, shift_15min, global_mean)
        
        it_powers_h = self.to_hourly(it_powers)
        it_baselines_h = self.to_hourly(it_baselines)
        opt_h = self.optimize_load(it_powers_h, shift_hourly, global_mean)

        self.results = {
            "peak_day": peak_day,
            "global_mean": global_mean,
            "it_powers": it_powers,
            "it_powers_h": it_powers_h,
            "opt_15": opt_15,
            "opt_h": opt_h,
            "it_baselines": it_baselines,
            "it_baselines_h": it_baselines_h
        }
        return self.results

    def plot_results(self):
        """Generates the analysis dashboard."""
        res = self.results
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # Subplot logic (Summarized for brevity)
        # Using a helper for repeated plot logic is cleaner
        def _style_ax(ax, title, ylabel, xlabel="", ylim=(0, 1.2e8)):
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_xlabel(xlabel, fontweight="bold")
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)

        # Plotting Original 15-min
        for i, (name, _) in enumerate(self.parameter_sets):
            axs[0, 0].plot(res["it_powers"][i], '--', label=name)
        _style_ax(axs[0, 0], "15-min: Original Power", "Watts")
        axs[0, 0].legend(ncol=2, fontsize=8)

        # Plotting Optimized 15-min
        for i, (name, _) in enumerate(self.parameter_sets):
            axs[0, 1].plot(res["opt_15"][i], label=f"{name} Opt")
        _style_ax(axs[0, 1], "15-min: Optimized", "Watts")

        # Error Plot (Simplified)
        err_before = np.mean(np.abs(res["it_powers"] - res["global_mean"]), axis=0)
        err_after = np.mean(np.abs(res["opt_15"] - res["global_mean"]), axis=0)
        axs[0, 2].plot(err_before, label="Before")
        axs[0, 2].plot(err_after, '--', label="After")
        _style_ax(axs[0, 2], "15-min: Mean Abs Error", "Error (W)", ylim=(0, err_before.max()*1.2))
        axs[0, 2].legend()

        # Hourly
        for i, (name, _) in enumerate(self.parameter_sets):
            axs[1, 0].plot(res["it_powers_h"][i], '--', label=name)
        _style_ax(axs[1, 0], "Hourly: Original Power", "Watts")
        axs[1, 0].legend(ncol=2, fontsize=8)

        # Plotting Optimized 15-min
        for i, (name, _) in enumerate(self.parameter_sets):
            axs[1, 1].plot(res["opt_h"][i], label=f"{name} Opt")
        _style_ax(axs[1, 1], "Hourly: Optimized", "Watts")

        # Error Plot (Simplified)
        err_before = np.mean(np.abs(res["it_powers_h"] - res["global_mean"]), axis=0)
        err_after = np.mean(np.abs(res["opt_h"] - res["global_mean"]), axis=0)
        axs[1, 2].plot(err_before, label="Before")
        axs[1, 2].plot(err_after, '--', label="After")
        _style_ax(axs[1, 2], "Hourly: Mean Abs Error", "Error (W)", ylim=(0, err_before.max()*1.2))
        axs[1, 2].legend()

        plt.tight_layout()
        plt.show()

# ----------------------- Metrics Utilities -----------------------

def get_peak_reduction(original: np.ndarray, optimized: np.ndarray) -> float:
    """
    Calculates the average daily peak reduction over a 365-day period.
    """
    daily_peaks_orig = np.max(original)
    daily_peaks_opt = np.max(optimized)
    peak_red = (daily_peaks_orig - daily_peaks_opt) / daily_peaks_orig
    return np.mean(peak_red)

# ----------------------- Execution -----------------------

if __name__ == "__main__":
    # Parameter Set Configuration Sample
    params = [
        ("P1", (0.7, 1.0, 12, 20, 3.5, 3.5, 0.15, 0.85, 0.7, 1e8)),
        ("P2", (1.0, 1.3, 6, 14, 3, 4, 0.15, 0.85, 0.7, 1e8)),
        ("P3", (0.7, 1.0, 10, 18, 4, 3, 0.15, 0.85, 0.7, 1e8)),
        ("P4", (0.7, 1.0, 12, 20, 2.0, 2.0, 0.15, 0.85, 0.8, 1e8)),
        ("P5", (0.7, 1.0, 12, 20, 3.5, 3.5, 0.15, 0.90, 0.9, 1e8)),
    ]

    optimizer = WorkloadOptimizer(params, peak_threshold=IT_CAPACITY)
    
    print("Starting Analysis...")
    analysis_data = optimizer.analyze(shift_15min=1.0, shift_hourly=1.0)
    
    # Calculate Summary Stats
    red_15 = get_peak_reduction(analysis_data["it_powers"], analysis_data["opt_15"])
    red_h = get_peak_reduction(analysis_data["it_powers"], analysis_data["opt_h"])
    print(f"Analysis Complete")
    print(f"15-min - Peak reduction (after optimization): -{red_15*100:.2f}% ")
    print(f"Hourly - Peak reduction (after optimization): -{red_h*100:.2f}% ")
    
    optimizer.plot_results()