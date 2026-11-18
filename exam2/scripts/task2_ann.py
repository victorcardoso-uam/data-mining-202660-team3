"""Task 2: MLPRegressor, comparison with Task 1 models, save the winning model."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common import FEATURES, FIGS, MODELS, REPORTS, TARGET, load_data, metrics, split

MODELS.mkdir(parents=True, exist_ok=True)
df = load_data()
X_train, X_test, y_train, y_test = split(df)

scaler = StandardScaler().fit(X_train)          # fit only on train
Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", early_stopping=True,
                   max_iter=1000, random_state=42).fit(Xtr_s, y_train)
m_mlp = metrics(y_test, mlp.predict(Xte_s), p=len(FEATURES))
print(f"MLP stopped after {mlp.n_iter_} epochs; best validation score (R2) = {mlp.best_validation_score_:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, color="k")
plt.yscale("log")
plt.xlabel("Epoch"); plt.ylabel("Training loss (squared error / 2, log scale)")
plt.title("Task 2 - MLPRegressor (64, 32) loss curve")
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(FIGS / "task2_loss_curve.png", dpi=130)

# Refit Task 1 models (same split) so every model object is available to save
ols = LinearRegression().fit(X_train, y_train)
ols_log = LinearRegression().fit(X_train, np.log(y_train))
poly = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                 ("lr", LinearRegression())]).fit(X_train, y_train)
p_poly = poly.named_steps["poly"].n_output_features_

candidates = {
    "OLS Linear": (ols, metrics(y_test, ols.predict(X_test), 5), False, False),
    "OLS Log-Target": (ols_log, metrics(y_test, np.exp(ols_log.predict(X_test)), 5), False, True),
    "Polynomial (deg 2)": (poly, metrics(y_test, poly.predict(X_test), p_poly), False, False),
    "MLP (64, 32)": (mlp, m_mlp, True, False),
}
comp = pd.DataFrame({k: v[1] for k, v in candidates.items()}).T
comp.index.name = "Model"
comp.to_csv(REPORTS / "model_comparison.csv", float_format="%.4f")
print("\n=== TASK 2: Model comparison (test set) ===")
print(comp.round(4).to_string())

winner = comp["RMSE"].idxmin()
model, m, needs_scaler, log_t = candidates[winner]
joblib.dump(model, MODELS / "best_model.joblib")
joblib.dump(scaler, MODELS / "scaler.joblib")
if winner != "MLP (64, 32)":
    joblib.dump(mlp, MODELS / "mlp_model.joblib")

meta = {
    "features": FEATURES, "target": TARGET, "winning_model": winner,
    "requires_scaler": needs_scaler, "log_target": log_t,
    "metrics": {k: {kk: round(float(vv), 4) for kk, vv in v[1].items()} for k, v in candidates.items()},
    "mlp": {"hidden_layer_sizes": [64, 32], "activation": "relu", "early_stopping": True,
            "max_iter": 1000, "n_iter": int(mlp.n_iter_), "requires_scaler": True},
    "split": {"test_size": 0.2, "random_state": 42},
}
(MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))
print(f"\n>>> WINNING MODEL (lowest test RMSE): {winner}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")
print(f"    MLP RMSE={m_mlp['RMSE']:.4f}. Saved best_model.joblib, scaler.joblib"
      + (", mlp_model.joblib" if winner != "MLP (64, 32)" else "") + ", metadata.json")
