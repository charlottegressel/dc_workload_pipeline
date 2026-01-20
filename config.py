"""
Configuration settings for the AI Data Center Flexible Load Simulation.
Defines paths, IT workload profiles, OU process noise parameters, and cooling benchmarks.
"""

from pathlib import Path

# --- Project Directory Structure ---
# Automatically finds the project root based on this file's location
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "datasets"
OUTPUT_DIR = ROOT_DIR / "output"
FIGURE_DIR = ROOT_DIR / "figures"

# --- Simulation Time Parameters ---
DELTA_T_MINUTES = 15
INTERVALS_PER_HOUR = 60 // DELTA_T_MINUTES
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365  # Standard year logic (Leap day excluded)

# Total steps for a 365-day simulation (35,040 intervals)
TOTAL_STEPS = DAYS_PER_YEAR * HOURS_PER_DAY * INTERVALS_PER_HOUR

START_DATE = "2025-01-01 00:00:00"
END_DATE = "2025-12-31 23:45:00" #2025 has no leap day

# --- Workload Profile (Diurnal Sinusoidal Baselines) ---
# Parameters for peak amplitudes and mean-reversion timings
AMP1, AMP2 = 0.7, 1.0        # Amplitude factors
MU1, MU2 = 12, 20            # Peak hours (Noon and Evening)
SIGMA1, SIGMA2 = 3.5, 3.5    # Peak widths (Standard deviation in hours)

# --- IT Infrastructure Parameters ---
IT_CAPACITY = 100_000_000    # Total capacity (W or relevant units)
POW_SERVER = 4200            # Power draw per server (W)
BASELINE_FLOOR = 0.3         # Minimum utilization floor
BASELINE_CAP = 0.9           # Maximum utilization cap
INF_SHARE = 0.7              # Share of load dedicated to Inference

# --- Ornstein-Uhlenbeck (OU) Noise Processes ---
# Models the stochastic nature of AI workloads
OU_DT = DELTA_T_MINUTES / 60   # Time step in hours
OU_THETA_INF = 0.25            # Mean-reversion strength (Inference)
OU_SIGMA_INF = 0.04            # Noise volatility (Inference)
OU_THETA_TR = 1.00             # Mean-reversion strength (Training)
OU_SIGMA_TR = 0.02             # Noise volatility (Training)

# --- Cooling & Environmental Parameters ---
CITY = "newyork"               # Default location
COOLING_COP = 11.7             # Liquid cooling Coefficient of Performance
COOLING_REF_TEMP = 23          # Target indoor setpoint (°C)

# Server/Rack physical parameters
COOLING_H_S_A_S = 259.35       # Heat transfer coefficient per server
COOLING_A = 0.43 / 0.4         # Surface area ratio
COOLING_U_POS = 0.35           # Positive thermal gain coefficient
COOLING_U_NEG = 25             # Negative thermal loss coefficient

# --- Dataset File Mapping ---
BIOHPC_FILE = DATA_DIR / "cbsugpu08_usage_stats.txt"

# --- Initialization ---
# Ensure necessary directories exist to prevent FileNotFoundError
for folder in [OUTPUT_DIR, FIGURE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
