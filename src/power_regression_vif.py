"""
Activity 15 - Multiple Linear Regression, Multicollinearity & VIF Pruning
IIND4417 Data Mining & Modern AI Systems - Team 3
Dataset : data/raw/mlr_team3_ev_battery.csv  (EV Battery Pack Thermodynamics)
Target  : heat_generation_w  [W]

Run from the repo root:
    python src/power_regression_vif.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save figures without opening a GUI window
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.nonparametric.smoothers_lowess import lowess

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "mlr_team3_ev_battery.csv"
FIG_PATH = ROOT / "reports" / "figures" / "regression_residuals.png"

TARGET = "heat_generation_w"
VIF_THRESHOLD = 3.0
MAX_PRUNE_STEPS = 2  # two-step stepwise cascade required by the activity

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF for every predictor. The constant is added before computing VIFs
    (otherwise statsmodels returns uncentered, inflated VIFs) and is then
    excluded from the table."""
    Xc = sm.add_constant(X, has_constant="add")
    rows = [
        (col, variance_inflation_factor(Xc.values, i))
        for i, col in enumerate(Xc.columns)
        if col != "const"
    ]
    return (
        pd.DataFrame(rows, columns=["feature", "VIF"])
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def fit_ols(y: pd.Series, X: pd.DataFrame):
    """OLS with an explicit intercept (sm.add_constant is mandatory)."""
    X_with_const = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_with_const).fit()


def coef_table(model) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "coef": model.params,
            "std_err": model.bse,
            "t": model.tvalues,
            "p_value": model.pvalues,
        }
    )


# ----------------------------------------------------------------------------
# Task 2 - Data ingestion & full OLS model
# ----------------------------------------------------------------------------
banner("TASK 2 | DATA INGESTION & FULL 6-FEATURE OLS MODEL")

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {DATA_PATH.relative_to(ROOT)}  shape={df.shape}")
print(f"Columns: {list(df.columns)}")

if TARGET not in df.columns:
    raise KeyError(f"Target '{TARGET}' not found. Columns: {list(df.columns)}")

# Continuous predictors = every numeric column except the target
X_full = df.drop(columns=[TARGET]).select_dtypes(include=np.number)
y = df[TARGET]

# Drop rows with missing values in any modelling column
mask = X_full.notna().all(axis=1) & y.notna()
if (~mask).sum():
    print(f"Dropping {(~mask).sum()} rows with missing values")
X_full, y = X_full[mask], y[mask]

print(f"Predictors ({X_full.shape[1]}): {list(X_full.columns)}")
if X_full.shape[1] != 6:
    print("WARNING: activity expects 6 predictors - check for ID/index columns.")

print("\nDescriptive statistics:")
print(df[list(X_full.columns) + [TARGET]].describe().T.round(4))

model_full = fit_ols(y, X_full)
print("\n" + str(model_full.summary()))

print("\nKey metrics (full model):")
print(f"  R-squared          = {model_full.rsquared:.4f}")
print(f"  Adjusted R-squared = {model_full.rsquared_adj:.4f}")
print(f"  F-statistic        = {model_full.fvalue:.4f}")
print(f"  F-statistic p-value= {model_full.f_pvalue:.4e}")
print("\nCoefficients, standard errors & p-values (full model):")
print(coef_table(model_full).round(6))

# ----------------------------------------------------------------------------
# Task 3 - Initial VIF
# ----------------------------------------------------------------------------
banner("TASK 3 | INITIAL VIF TABLE (6 FEATURES)")
vif_initial = compute_vif(X_full)
print(vif_initial.round(4).to_string(index=False))

print("\nPearson correlation matrix of predictors (supports the AI audit):")
print(X_full.corr().round(3))

# ----------------------------------------------------------------------------
# Task 4 - Stepwise VIF pruning (one feature per step, VIFs recomputed)
# ----------------------------------------------------------------------------
banner("TASK 4 | STEPWISE VIF PRUNING CASCADE (threshold VIF < 3.0)")
X_pruned = X_full.copy()
dropped = []

for step in range(1, MAX_PRUNE_STEPS + 1):
    vif_now = compute_vif(X_pruned)
    worst_feat, worst_vif = vif_now.loc[0, "feature"], vif_now.loc[0, "VIF"]
    if worst_vif < VIF_THRESHOLD:
        print(f"Step {step}: all VIFs already < {VIF_THRESHOLD}; stopping.")
        break
    X_pruned = X_pruned.drop(columns=[worst_feat])
    dropped.append((worst_feat, worst_vif))
    print(f"\nStep {step} | Dropped: {worst_feat}  (highest VIF = {worst_vif:.4f})")
    print(f"Recomputed VIFs for remaining {X_pruned.shape[1]} features:")
    print(compute_vif(X_pruned).round(4).to_string(index=False))

vif_final = compute_vif(X_pruned)
print("\nFinal clean feature VIFs:")
print(vif_final.round(4).to_string(index=False))
ok = (vif_final["VIF"] < VIF_THRESHOLD).all()
print(f"\nAll VIF < {VIF_THRESHOLD}? {'YES' if ok else 'NO - review pruning'}")

# ----------------------------------------------------------------------------
# Task 5 - Pruned model & comparison
# ----------------------------------------------------------------------------
banner(f"TASK 5 | PRUNED {X_pruned.shape[1]}-FEATURE OLS MODEL & COMPARISON")
model_pruned = fit_ols(y, X_pruned)
print(model_pruned.summary())

shared = list(X_pruned.columns)
# Dominant feature = retained feature with the largest |t| in the pruned model
dominant = model_pruned.tvalues.drop("const").abs().idxmax()

comparison = pd.DataFrame(
    {
        "Metric": [
            "Model R-squared",
            "Adjusted R-squared",
            "Number of Features",
            f"Dominant Feature StdErr ({dominant})",
            "F-Statistic p-value",
        ],
        f"Full OLS ({X_full.shape[1]} features)": [
            f"{model_full.rsquared:.4f}",
            f"{model_full.rsquared_adj:.4f}",
            X_full.shape[1],
            f"{model_full.bse[dominant]:.6f}",
            f"{model_full.f_pvalue:.4e}",
        ],
        f"Pruned OLS ({X_pruned.shape[1]} features)": [
            f"{model_pruned.rsquared:.4f}",
            f"{model_pruned.rsquared_adj:.4f}",
            X_pruned.shape[1],
            f"{model_pruned.bse[dominant]:.6f}",
            f"{model_pruned.f_pvalue:.4e}",
        ],
    }
)
print("\nSide-by-side comparison:")
print(comparison.to_string(index=False))

se_compare = pd.DataFrame(
    {
        "coef_full": model_full.params[["const"] + shared],
        "coef_pruned": model_pruned.params[["const"] + shared],
        "SE_full": model_full.bse[["const"] + shared],
        "SE_pruned": model_pruned.bse[["const"] + shared],
    }
)
se_compare["SE_reduction_%"] = (1 - se_compare["SE_pruned"] / se_compare["SE_full"]) * 100
print("\nCoefficient & standard-error stability (retained features):")
print(se_compare.round(6))
print(f"\nDropped features: {[f for f, _ in dropped]}")
print(f"R-squared change: {model_pruned.rsquared - model_full.rsquared:+.4f}")

# ----------------------------------------------------------------------------
# Task 6 - Residual diagnostics
# ----------------------------------------------------------------------------
banner("TASK 6 | RESIDUAL DIAGNOSTICS (PRUNED MODEL)")
fitted = model_pruned.fittedvalues
resid = model_pruned.resid

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1 - Residuals vs Fitted
ax = axes[0]
ax.scatter(fitted, resid, s=14, alpha=0.6, edgecolor="none")
ax.axhline(0, color="red", linestyle="--", linewidth=1.2, label="Zero residual")
smooth = lowess(resid, fitted, frac=0.3)
ax.plot(smooth[:, 0], smooth[:, 1], color="black", linewidth=1.5, label="LOWESS trend")
ax.set_title("Residuals vs. Fitted Values")
ax.set_xlabel("Fitted heat_generation_w [W]")
ax.set_ylabel("Residual [W]")
ax.legend()
ax.grid(alpha=0.3)

# Panel 2 - Normal Q-Q
ax = axes[1]
sm.qqplot(resid, line="s", ax=ax, markersize=4, alpha=0.6)
ax.set_title("Normal Q-Q Plot of Residuals")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles [W]")
ax.grid(alpha=0.3)

fig.suptitle(
    f"Team 3 EV Battery - Pruned OLS ({X_pruned.shape[1]} features) Residual Diagnostics",
    fontsize=13,
)
fig.tight_layout()
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=150)
plt.close(fig)
print(f"Saved figure: {FIG_PATH.relative_to(ROOT)}")

# Objective support for the written assessment
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(resid, sm.add_constant(X_pruned))
jb = sm.stats.jarque_bera(resid)
print(f"\nResidual mean                 = {resid.mean():.6e}")
print(f"Residual std                  = {resid.std():.6f}")
print(f"Breusch-Pagan LM p-value      = {bp_lm_p:.4f}  (> 0.05 -> constant variance)")
print(f"Jarque-Bera p-value           = {jb[1]:.4f}  (> 0.05 -> normal residuals)")
print(f"Residual skew / kurtosis      = {jb[2]:.4f} / {jb[3]:.4f}")
print("\nDone.")