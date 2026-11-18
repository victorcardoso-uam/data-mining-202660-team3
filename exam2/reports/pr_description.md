## Exam 2 Part 2: Team 3 EV Battery

Dataset: `exam2_team3_ev_battery.csv` (600 rows, no nulls) | Target: `battery_cell_temp_c`
Split: 80/20, `random_state=42`. All code is in `exam2/`.

### Task 1: Regression and residual diagnostics (`scripts/task1_regression.py`)
- OLS base, OLS with log target (back-transformed with `exp`), polynomial degree 2.
- 2x2 residual dashboard (Residuals vs Fitted, Q-Q, Scale-Location, Leverage + Cook's) in `reports/figures/task1_residual_dashboard.png`.
- Residuals look linear, roughly normal (Shapiro p = 0.20), homoscedastic (Breusch-Pagan p = 0.60), no influential points (max Cook's D = 0.031).

### Task 2: ANN (`scripts/task2_ann.py`)
- `MLPRegressor((64, 32), relu, early_stopping=True, max_iter=1000)` with StandardScaler fit on train only; loss curve in `reports/figures/task2_loss_curve.png`.

| Model | R2 | Adj R2 | MAE | RMSE |
|---|---|---|---|---|
| **OLS Linear (winner)** | **0.9766** | **0.9756** | **1.2131** | **1.5659** |
| OLS Log-Target | 0.9499 | 0.9477 | 1.6605 | 2.2935 |
| Polynomial (deg 2) | 0.9759 | 0.9710 | 1.2394 | 1.5899 |
| MLP (64, 32) | 0.9688 | 0.9674 | 1.3791 | 1.8106 |

Winning model (lowest test RMSE): **OLS Linear** -> `models/best_model.joblib`; MLP in `models/mlp_model.joblib`, scaler in `models/scaler.joblib`, `models/metadata.json`.

### Task 3: Streamlit app (`app.py`)
- 4 tabs: Data Overview, Model Comparison, ANN Training, Real-Time What-If Scenario Inference.
- Alert policy: active cooling at 53 C (P75 = 52.56 C), critical at 59 C (P90 = 58.52 C).
- Presets checked with the real model: Normal -> 25.72 C (normal), High-risk -> 71.34 C (critical).
- Run: `cd exam2 && streamlit run app.py`. Screenshots in `reports/screenshots/`.

### Report
`reports/Exam2_Team3_Report.pdf` (regenerate with `python scripts/build_report.py`).
