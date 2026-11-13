"""
Activity 21 - ANN Architecture, Hyperparameter Optimization & Production Export
Team 3 - EV Battery | Target: battery_cell_temp_c
Course: Data Mining & Modern AI Systems (IIND4417)
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TEAM = 3
RANDOM_STATE = 42
CSV_PATH = "data/raw/ann_team3_ev_battery.csv"
TARGET = "battery_cell_temp_c"

# ------------------------------------------------------------------
# 0. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum(), "\n")

X = df.drop(columns=[TARGET])
y = df[TARGET]
n_features = X.shape[1]

# ------------------------------------------------------------------
# STEP 1. SPLIT (80/20) + FEATURE SCALING (no data leakage)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_s = scaler.transform(X_test)         # transform test with train stats

print("Scaled train mean (~0):", np.round(X_train_s.mean(axis=0), 3))
print("Scaled train std  (~1):", np.round(X_train_s.std(axis=0), 3), "\n")

# ------------------------------------------------------------------
# STEP 2. DEFINE & TRAIN THE 5 ANN MODELS
# ------------------------------------------------------------------
configs = {
    "Model 1 (Shallow Baseline)":  dict(hidden_layer_sizes=(16,),         activation="relu", solver="adam"),
    "Model 2 (Single Wide Layer)": dict(hidden_layer_sizes=(64,),         activation="relu", solver="adam"),
    "Model 3 (Deep 2-Layer Net)":  dict(hidden_layer_sizes=(64, 32),      activation="relu", solver="adam"),
    "Model 4 (Deep Tanh)":         dict(hidden_layer_sizes=(128, 64, 32), activation="tanh", solver="adam"),
    "Model 5 (Regularized Early-Stop)": dict(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                                             alpha=0.01, early_stopping=True,
                                             validation_fraction=0.1, n_iter_no_change=20),
}

def count_params(model):
    return sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

models, results = {}, []
n_test = len(y_test)

for name, cfg in configs.items():
    mdl = MLPRegressor(max_iter=2000, random_state=RANDOM_STATE, **cfg)
    mdl.fit(X_train_s, y_train)
    models[name] = mdl

    pred_tr = mdl.predict(X_train_s)
    pred_te = mdl.predict(X_test_s)
    r2_te = r2_score(y_test, pred_te)

    results.append({
        "Model": name,
        "Params": count_params(mdl),
        "Epochs": mdl.n_iter_,
        "Train R2": r2_score(y_train, pred_tr),
        "Test R2": r2_te,
        "Adj R2": adjusted_r2(r2_te, n_test, n_features),
        "MAE": mean_absolute_error(y_test, pred_te),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_te)),
    })

# ------------------------------------------------------------------
# STEP 3. BENCHMARK TABLE
# ------------------------------------------------------------------
bench = pd.DataFrame(results).set_index("Model").round(4)
print("=== MULTI-MODEL BENCHMARK (holdout 20%) ===")
print(bench.to_string(), "\n")
bench.to_csv("ann_benchmark_team3.csv")

best_name = bench["Test R2"].idxmax()
best_model = models[best_name]
print(f"WINNING MODEL: {best_name} (Test R2 = {bench.loc[best_name, 'Test R2']})\n")

# ------------------------------------------------------------------
# STEP 4. DIAGNOSTIC PLOTS (2 panels)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: loss curves
for name, mdl in models.items():
    axes[0].plot(mdl.loss_curve_, label=name, linewidth=1.8)
axes[0].set_yscale("log")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training loss (log scale)")
axes[0].set_title("Panel A: Loss Convergence Curves (loss_curve_)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# Panel B: predicted vs actual (winning model, test set)
pred_best = best_model.predict(X_test_s)
axes[1].scatter(y_test, pred_best, alpha=0.7, edgecolor="k", linewidth=0.3)
lims = [min(y_test.min(), pred_best.min()), max(y_test.max(), pred_best.max())]
axes[1].plot(lims, lims, "r--", label="Perfect prediction (y = x)")
axes[1].set_xlabel("Actual battery_cell_temp_c (°C)")
axes[1].set_ylabel("Predicted battery_cell_temp_c (°C)")
axes[1].set_title(f"Panel B: Predicted vs Actual - {best_name}\n"
                  f"Test R² = {bench.loc[best_name, 'Test R2']:.4f}")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ann_diagnostics_team3.png", dpi=200)
plt.show()

# ------------------------------------------------------------------
# STEP 5. PRODUCTION SERIALIZATION
# ------------------------------------------------------------------
joblib.dump(best_model, f"best_ann_model_team{TEAM}.joblib")
joblib.dump(scaler, f"ann_scaler_team{TEAM}.joblib")
print(f"Saved: best_ann_model_team{TEAM}.joblib, ann_scaler_team{TEAM}.joblib")

# Sanity check: reload and predict
m = joblib.load(f"best_ann_model_team{TEAM}.joblib")
s = joblib.load(f"ann_scaler_team{TEAM}.joblib")
check = m.predict(s.transform(X_test.iloc[:3]))
print("Reload check -> predicted:", np.round(check, 2), "| actual:", y_test.iloc[:3].values)

# ------------------------------------------------------------------
# EXTRA (for the report): why unscaled data hurts the ANN
# ------------------------------------------------------------------
raw = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                   max_iter=2000, random_state=RANDOM_STATE).fit(X_train, y_train)
print("\n[Unscaled demo] Test R2 without scaling:", round(r2_score(y_test, raw.predict(X_test)), 4),
      "| Scaled (Model 3):", bench.loc['Model 3 (Deep 2-Layer Net)', 'Test R2'])
print("Feature ranges (raw):\n", X.describe().loc[["min", "max"]].T.round(2))