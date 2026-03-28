# Physics-Informed ML for Air Temperature Prediction 🌡️

This project implements a **Physics-Guided Machine Learning** approach to predict daily air temperature. Instead of relying solely on historical patterns, this model incorporates the **Surface Energy Balance (SEB)** equation to improve accuracy and physical consistency.

## Overview
Traditional ML models often treat meteorological variables as independent numbers. This project uses **Physics-Guided Feature Engineering** to inject domain knowledge into Gradient Boosting and Random Forest models.

### The Physics: Surface Energy Balance
We approximate the following physical drivers from raw NASA POWER data:
* **Net Radiation ($R_n$):** The total energy available at the surface.
* **Vapor Pressure Deficit (VPD):** The "drying power" of the air, a proxy for Latent Heat Flux.
* **Sensible Heat Proxy ($H_{proxy}$):** Modeling heat exchange between the surface and air using wind speed and temperature gradients.
* **Energy Imbalance:** A feature representing the residual energy, forcing the model to recognize thermodynamic inconsistencies.

## Key Results
By adding physical proxies, we achieved a significant reduction in error compared to pure data-driven baselines.

| Model | RMSE (Lower is Better) | $R^2$ (Higher is Better) |
| :--- | :--- | :--- |
| **Pure GBM (Baseline)** | 1.533 | 0.945 |
| **Physics-Informed GBM** | **1.169** | **0.968** |

**Improvement:** The Physics-Informed Gradient Boosting model reduced prediction error by **~24%**.

### Setup 
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt
python backend/fetch_data.py
python backend/preprocess.py
python backend train_model.py

