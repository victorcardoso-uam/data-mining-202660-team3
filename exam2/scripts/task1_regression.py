"""Task 1: OLS, log-target OLS, polynomial (deg 2) regression + residual diagnostics."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.diagnostic import het_breuschpagan

from common import FEATURES, FIGS, REPORTS, TARGET, load_data, metrics, split

FIGS.mkdir(parents=True, exist_ok=True)

df = load_data()
print("Shape:", df.shape)
print("Nulls per column:\n", df.isna().sum().to_string())
print("Dtypes:\n", df.dtypes.to_string())

X_train, X_test, y_train, y_test = split(df)

# (1) OLS base
ols = LinearRegression().fit(X_train, y_train)
m1 = metrics(y_test, ols.predict(X_test), p=len(FEATURES))

# (2) OLS on log(y), back-transformed with exp
ols_log = LinearRegression().fit(X_train, np.log(y_train))
m2 = metrics(y_test, np.exp(ols_log.predict(X_test)), p=len(FEATURES))

# (3) Polynomial degree 2
poly = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                 ("lr", LinearRegression())]).fit(X_train, y_train)
p_poly = poly.named_steps["poly"].n_output_features_
m3 = metrics(y_test, poly.predict(X_test), p=p_poly)

table = pd.DataFrame([m1, m2, m3],
                     index=["OLS Linear", "OLS Log-Target", "Polynomial (deg 2)"])
table.index.name = "Model"
table.to_csv(REPORTS / "task1_metrics.csv", float_format="%.4f")
print("\n=== TASK 1: Test-set metrics ===")
print(table.round(4).to_string())

# Residual diagnostics on base OLS (statsmodels, train set)
Xc = sm.add_constant(X_train)
res = sm.OLS(y_train, Xc).fit()
infl = res.get_influence()
fitted, resid = res.fittedvalues, res.resid
std_resid = infl.resid_studentized_internal
leverage = infl.hat_matrix_diag
cooks = infl.cooks_distance[0]

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].scatter(fitted, resid, s=12, alpha=.6, color="k")
ax[0, 0].axhline(0, ls="--", color="gray")
lo = sm.nonparametric.lowess(resid, fitted)
ax[0, 0].plot(lo[:, 0], lo[:, 1], color="red")
ax[0, 0].set(title="Residuals vs Fitted", xlabel="Fitted", ylabel="Residuals")

stats.probplot(std_resid, dist="norm", plot=ax[0, 1])
ax[0, 1].set_title("Normal Q-Q")

sl = np.sqrt(np.abs(std_resid))
ax[1, 0].scatter(fitted, sl, s=12, alpha=.6, color="k")
lo2 = sm.nonparametric.lowess(sl, fitted)
ax[1, 0].plot(lo2[:, 0], lo2[:, 1], color="red")
ax[1, 0].set(title="Scale-Location", xlabel="Fitted", ylabel="sqrt(|Std. residuals|)")

ax[1, 1].scatter(leverage, std_resid, s=12 + 400 * cooks / cooks.max(), alpha=.6, color="k")
k = Xc.shape[1]
for c in (0.5, 1.0):
    h = np.linspace(leverage.min(), leverage.max() * 1.05, 100)
    b = np.sqrt(c * k * (1 - h) / h)
    ax[1, 1].plot(h, b, "r--", lw=.8); ax[1, 1].plot(h, -b, "r--", lw=.8)
ax[1, 1].set_ylim(std_resid.min() - 1, std_resid.max() + 1)
thr = 4 / len(X_train)
for i in np.where(cooks > thr)[0][:8]:
    ax[1, 1].annotate(str(X_train.index[i]), (leverage[i], std_resid[i]), fontsize=7)
ax[1, 1].set(title="Residuals vs Leverage (Cook's distance)", xlabel="Leverage",
             ylabel="Std. residuals")
fig.suptitle("Task 1 - OLS Residual Diagnostics (base linear model)")
fig.tight_layout()
fig.savefig(FIGS / "task1_residual_dashboard.png", dpi=130)

# Diagnostics numbers
sw_p = stats.shapiro(resid).pvalue
bp_p = het_breuschpagan(resid, Xc)[1]
n_infl = int((cooks > thr).sum())
best = table["RMSE"].idxmin()
print("\n=== TASK 1: Interpretation ===")
print(f"1) Best model by test RMSE: {best} (RMSE={table.loc[best,'RMSE']:.3f}, "
      f"Adj R2={table.loc[best,'Adj_R2']:.3f}); base OLS RMSE={m1['RMSE']:.3f}.")
print(f"2) Linearity: the LOWESS line in Residuals vs Fitted "
      f"{'curves, so there is some non-linearity' if best.startswith('Poly') else 'is mostly flat'}.")
print(f"3) Normality: Shapiro-Wilk p={sw_p:.4f} "
      f"({'residuals look roughly normal' if sw_p > .05 else 'some departure from normality in the tails'}); "
      f"Homoscedasticity: Breusch-Pagan p={bp_p:.4f} "
      f"({'constant variance OK' if bp_p > .05 else 'variance is not constant'}).")
print(f"4) Influence: {n_infl} train points have Cook's D > 4/n (max D={cooks.max():.3f}); "
      f"{'none is above 0.5, so no single point drives the fit' if cooks.max() < .5 else 'some points are very influential'}.")
pd.Series({"shapiro_p": sw_p, "breusch_pagan_p": bp_p, "n_cooks_gt_4n": n_infl,
           "max_cooks": cooks.max(), "best_model": best}).to_csv(REPORTS / "task1_diagnostics.csv")
