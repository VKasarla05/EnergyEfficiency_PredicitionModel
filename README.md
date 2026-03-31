# Energy-Efficient Deep Learning — Predictive Modeling of GPU/CPU Energy Consumption

<img width="731" height="447" alt="image" src="https://github.com/user-attachments/assets/e4043da9-53fe-4325-8587-e1dc1d989315" />

> Predicting how much energy a deep learning training job will consume **before it runs** — enabling smarter scheduling, reduced costs, and lower carbon emissions.

---

## The Problem

Modern deep learning training jobs run on GPU/CPU clusters for hours or days — but researchers have **no reliable way to estimate energy consumption before launching a job**. This leads to:

- Inefficient compute scheduling
- Unpredictable resource costs
- Unnecessary carbon emissions
- Poor sustainability planning

This project solves that by building predictive models that estimate total energy consumption from job-level workload characteristics — **before execution**.

---

## Results at a Glance

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| **Neural Network (avNNet)** | **0.0498** | **0.0233** | **0.9974** |
| SVM Radial | 0.0591 | 0.0461 | 0.9965 |
| MARS | — | — | 0.993 (train) |
| KNN | — | — | 0.976 (train) |
| OLS (baseline) | — | — | 0.944 (train) |

**Neural Network achieved R² = 0.9974** on unseen test data — near-perfect generalization, outperforming all 8 competing models including SVM, MARS, Ridge, Lasso, Elastic Net, PLS, and KNN.

---

## Dataset

**BUTTER-E: Energy Consumption for Deep Learning Workloads**  
Published by the **National Renewable Energy Laboratory (NREL)**

- **Source:** [https://data.openei.org/submissions/5991](https://data.openei.org/submissions/5991)
- **Size:** 29,647 observations · 33 predictors
- **Target:** `energy` — total workload energy consumption in joules

### Predictor Types

**Continuous (20):** runtime, batch size, learning rate, power, CPU energy features, model depth/size, overhead statistics

**Categorical (13):** model shape, dataset type, optimizer, scheduling day/hour, filter configurations

---

## Methodology

### Data Preprocessing Pipeline

```
Raw Data (29,647 × 33)
    │
    ├── Timestamp Feature Engineering
    │     └── hour-of-day · day-of-week (scheduling patterns)
    │
    ├── Missing Value Treatment
    │     └── Zero missing values — no imputation needed
    │
    ├── Categorical Encoding (dummyVars)
    │     └── 33 → 56 predictors
    │
    ├── Near-Zero-Variance (NZV) Filtering
    │     └── 56 → 52 predictors
    │
    ├── Train / Test Split (80/20)
    │     └── Train: 29,647 · Test: 7,408
    │
    ├── Two Training Copies
    │     ├── Copy 1: Collinearity removed (cutoff 0.85) → for OLS, NNet, MARS
    │     └── Copy 2: All predictors retained → for Ridge, Lasso, ENet, SVM, KNN, PLS
    │
    ├── Box–Cox Transformation (skewness correction)
    ├── Centering & Scaling (standardization)
    ├── Spatial Sign Transformation (outlier resistance)
    └── Target Variable Scaling (prevent data leakage)
```

### Why Two Training Copies?

| Copy | Collinearity | Models |
|---|---|---|
| Copy 1 | Removed (threshold 0.85) | OLS · Neural Network · MARS |
| Copy 2 | Retained | Ridge · Lasso · Elastic Net · SVM · KNN · PLS |

Models like Ridge and SVM handle collinearity through regularization or kernel methods — removing correlated predictors would reduce their information. OLS and Neural Networks are sensitive to multicollinearity and benefit from a cleaner feature set.

### Model Tuning

All models tuned via **10-fold cross-validation** minimizing RMSE:

| Model | Tuned Parameters |
|---|---|
| Neural Network (avNNet) | Hidden units (3–9) · Weight decay (0.001–0.1) |
| MARS | Spline degree (1–2) · nprune (2–50) |
| SVM Radial | Cost C · Sigma |
| KNN | k = 1–20 |
| Ridge | Lambda (0–0.1) |
| Lasso | Fraction (0.05–1.0) |
| Elastic Net | Lambda × Fraction grid |
| PLS | Number of components (1–20) |

---

## Key Finding: Nonlinearity Confirmed

Linear models (OLS, Ridge, Lasso, Elastic Net, PLS) all plateaued around **R² ≈ 0.927** despite regularization and feature engineering. Nonlinear models (Neural Network, SVM, MARS) broke through to **R² > 0.993** — confirming that energy consumption in deep learning workloads is an **inherently nonlinear problem**.

---

## Top 10 Predictors (Neural Network)

| Rank | Predictor | Importance |
|---|---|---|
| 1 | `run_time` | 100.00 |
| 2 | `batch_energy_4_cpu_extra_shapes` | 36.83 |
| 3 | `batch_energy_2_cpu` | 36.27 |
| 4 | `shape_rectangle` | 36.10 |
| 5 | `shape_trapezoid` | 30.76 |
| 6 | `batch_energy_3_cpu_extra_depths` | 29.99 |
| 7 | `size` | 23.17 |
| 8 | `energy_overhead` | 19.40 |
| 9 | `start_wday_Sunday` | 18.61 |
| 10 | `update_wday_Thursday` | 16.21 |

**Insight:** Energy usage is driven by a combination of execution duration, CPU batch operations, model shape configuration, and scheduling day patterns — not just raw compute specs.

---

## Project Structure

```
energy-efficient-deeplearning-prediction/
├── Predictive_Report.pdf        # Full project report with all visuals
├── energy_prediction.R          # Complete R script (all steps)
├── README.md                    # This file
└── plots/                       # Generated visualizations (optional)
    ├── predictor_vs_energy.png
    ├── nzv_variance_plot.png
    ├── boxcox_before_after.png
    ├── correlation_before_after.png
    ├── ridge_tuning.png
    ├── lasso_tuning.png
    ├── enet_tuning.png
    ├── svm_tuning.png
    ├── nnet_tuning.png
    ├── mars_tuning.png
    ├── knn_tuning.png
    └── top10_predictors_nnet.png
```

---

## How to Run

### Prerequisites

```r
install.packages(c(
  "caret", "e1071", "earth", "ggplot2",
  "corrplot", "reshape2", "pls", "elasticnet"
))
```

### Steps

```r
# 1. Clone the repo
# 2. Set your dataset path in line 10 of energy_prediction.R
# 3. Source the script

source("energy_prediction.R")

# The script will:
# - Load and inspect the NREL dataset
# - Run the full preprocessing pipeline
# - Train all 9 models with 10-fold CV
# - Evaluate on the held-out test set
# - Output performance tables and plots
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | R |
| ML Framework | caret |
| Models | avNNet · svmRadial · earth (MARS) · knn · lm · ridge · lasso · enet · pls |
| Preprocessing | BoxCox · center/scale · spatialSign · nearZeroVar · dummyVars |
| Visualization | ggplot2 · corrplot · base R plots |
| Dataset | NREL BUTTER-E (Open Energy Data Initiative) |

---

## Authors

**Vyshnavi Priya Kasarla** · M.S. Data Science, Michigan Technological University  
**Madhava Narasimha Ajay Varma Penmatsa** · Michigan Technological University  
Course: MA5790 — Predictive Modeling

---

## References

1. Tripp et al. (2022). BUTTER-E Energy Dataset. NREL / OEDI. https://doi.org/10.25984/2329316
2. Strubell et al. (2019). Energy and Policy Considerations for Deep Learning in NLP. ACL 2019.
3. Patterson et al. (2021). Carbon Emissions and Large Neural Network Training. arXiv:2104.10350
4. Henderson et al. (2020). Systematic Reporting of Energy and Carbon Footprints of ML. JMLR 21(248).
5. Friedman (1991). Multivariate Adaptive Regression Splines. Annals of Statistics, 19(1).
6. Boser, Guyon & Vapnik (1992). Training Algorithm for Optimal Margin Classifiers. COLT 1992.

---

## License

This project is for academic and research purposes. Dataset is published under the Open Energy Data Initiative (OEDI) — see [NREL data terms](https://data.openei.org/submissions/5991) for usage details.
