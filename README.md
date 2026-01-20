# AI Data Center Workload Simulation Pipeline  
**Modeling AI Data Center Workloads for Power System Flexibility and Grid Stress Mitigation**

## Overview

This repository contains the simulation code accompanying the paper:

> **Modeling AI Data Center Workloads for Power System Flexibility and Grid Stress Mitigation**  
> *Charlotte Gressel, Richard Mahuze, K. Max Zhang — Cornell University*

The objective of this work is to model **AI data center (DC)** load profiles using empirically calibrated data and a comprehensive set of system parameters, at high temporal resolution and over daily operational timescales. The framework explicitly distinguishes between **inference and training** AI workloads, capturing their fundamentally different temporal characteristics and flexibility. These IT workloads are coupled with an adaptive **liquid-chiller cooling** model, enabling realistic representation of cooling demand and its interaction with workload dynamics.

The framework is intended to support grid operators, policymakers, and researchers in understanding and assessing in future simulations the impact of rapidly growing AI data center electricity demand on power systems, including peak demand, variability, and flexibility potential.

<img width="700" height="692" alt="Simulated Load Profile Breakdown (7 days sample)" src="https://github.com/user-attachments/assets/f0c0a548-5912-410d-85a9-77d3ccb0237b" />

---

## AI Data Center Simulation Tool

This repository provides an **open-source, scalable, and adaptable AI data center load profile simulation framework**, designed to be integrated into broader studies of:
- load aggregation,
- temporal and geographic load shifting,
- grid stress and peak demand analysis.

A key contribution of this model is its explicit focus on **AI-specific workloads**, distinguishing between:

- **Inference workloads**  
  Non-deferrable, highly variable, and driven by daily usage patterns, task diversity, and query stochasticity.

- **Training workloads**  
  Highly power-intensive but significantly more flexible and orchestrable in time and space.

The framework incorporates:
- stochastic variability via **Ornstein–Uhlenbeck mean-reverting processes**,  
- IT capacity maximization assumptions for training workload allocation,  
- cooling power consumption modeled through **thermodynamic relationships** and real-world temperature data.

---

## Core Modules

The repository includes the following main components:

- **`config.py`**  
  Centralized definition of all model parameters, assumptions, and directory paths.

- **`workload_simulation.py`**  
  Core simulation engine that generates IT utilization baselines, applies stochastic variability, and computes total power consumption.

- **`biohpc_regression.py`**  
  Empirical regression using BioHPC datasets to link GPU utilization to power consumption and calibrate model parameters.

- **`visualization.py`**  
  Visualization utilities for daily load profiles and statistical summaries.

---

## Applications

The `applications/` directory contains example scripts demonstrating how the model can be applied to reproduce and extend the analyses presented in the paper:

- **`app_timeshift.py`**  
  Applies a convex optimization framework to assess temporal load shifting strategies, with a focus on resolution effects (hourly vs. 15-minute).

- **`app_geoshift.py`**  
  Evaluates geographic load diversification across six U.S. time zones, quantifying peak demand reduction and daily variability (min–max ratios).

- **`app_param_sensiv.py`**  
  Sensitivity analysis of load profiles with respect to key workload and system parameters.

- **`app_infshare_sensiv.py`**  
  Sensitivity analysis of increasing inference workload share in total IT demand.

- **`app_pue.py`**  
  Analysis of PUE variation across time and seasons.

- **`app_coppue.py`**  
  Evaluation of the impact of increasing cooling efficiency (COP) on PUE.

- **`app_joballocalgo.py`**  
  Demonstrates a training job allocation algorithm that motivates the IT capacity maximization assumption used to model training load profiles.

Together, these applications illustrate:
- the importance of high-resolution load modeling,
- the impact of load shifting on peak demand and variability,
- and the role of cooling efficiency and its temporal dynamics.

All figures and numerical results presented in the paper can be reproduced using this repository.

---

## Scientific Contributions

This codebase supports the analyses presented in the paper, including:

- High-resolution stochastic modeling of AI inference and training workloads  
- Empirical calibration using Big Tech–style assumptions and BioHPC GPU utilization and power traces  
- Mapping IT utilization to total facility power using PUE/COP-based cooling models  
- Quantification of peak demand reduction through:
  - temporal shifting of deferrable workloads,
  - geographic diversification of data center locations  
- Monte Carlo–based statistical comparison of baseline and flexible scenarios  
- Sensitivity analysis across workload, infrastructure, and cooling parameters  

The framework is intended for **planning-level power system analysis**, not real-time operational control.

---

## Repository Structure

```bash
dc_workload_pipeline/
├── config.py                     # Global configuration parameters
├── workload_simulation.py        # Core simulation engine
├── biohpc_regression.py          # Empirical workload calibration
├── visualization.py              # Plotting utilities
│
├── applications/                 # Experiment scripts
│   ├── app_timeshift.py          # Temporal load shifting
│   ├── app_geoshift.py           # Geographic diversification
│   ├── app_pue.py                # PUE/COP sensitivity
│   ├── app_infshare_sensiv.py    # Inference share sensitivity
│   ├── app_param_sensiv.py       # Parameter sensitivity
│   └── app_joballocalgo.py       # Training job allocation example
│
├── datasets/                     # Input datasets
│   ├── biohpc/                   # BioHPC traces
│   └── nsrdb/                    # Weather data
│
├── figures/                      # Generated figures
├── output/                       # Numerical outputs
│
├── requirements.txt
└── README.md
```

To ensure that internal package dependencies and relative paths are resolved correctly, always execute the modules from the project root using the Python module flag (-m).

For the **core** modules: 

```bash
python -m dc_workload_pipeline.workload_simulation
```

For **applications**:

```bash
python -m dc_workload_pipeline.applications.app_timeshift
```
