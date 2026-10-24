import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("regression_team3_ev_battery.csv")
target = "battery_cell_temp_c"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

models = {
    "Model 1": ["ambient_temp_c"],
    "Model 2": ["ambient_temp_c", "discharge_current_a"],
    "Model 3": ["ambient_temp_c", "discharge_current_a", "coolant_flow_l_min"],
    "Model 4": ["ambient_temp_c", "discharge_current_a", "coolant_flow_l_min", "internal_resistance_mohm"],
    "Model 5": ["ambient_temp_c", "discharge_current_a", "coolant_flow_l_min", "internal_resistance_mohm", "state_of_charge_pct"],
}

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

rows = []
preds = {}
for name, feats in models.items():
    lr = LinearRegression().fit(X_train[feats], y_train)
    yp_train = lr.predict(X_train[feats])
    yp_test = lr.predict(X_test[feats])
    preds[name] = yp_test
    test_r2 = r2_score(y_test, yp_test)
    rows.append({
        "Model": name,
        "Features": ", ".join(feats),
        "Train R2": round(r2_score(y_train, yp_train), 4),
        "Test R2": round(test_r2, 4),
        "Adjusted R2": round(adjusted_r2(test_r2, len(y_test), len(feats)), 4),
        "MAE": round(mean_absolute_error(y_test, yp_test), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, yp_test)), 4),
    })

results = pd.DataFrame(rows)
print(results.to_string(index=False))
results.to_csv("activity16_results.csv", index=False)

top2 = results.sort_values("RMSE").head(2)["Model"].tolist()
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for i, name in enumerate(top2):
    resid = y_test - preds[name]
    axes[i, 0].scatter(preds[name], resid, alpha=0.6)
    axes[i, 0].axhline(0, color="red", linestyle="--")
    axes[i, 0].set_title(f"{name}: Fitted vs Residuals")
    axes[i, 0].set_xlabel("Fitted values (C)")
    axes[i, 0].set_ylabel("Residuals (C)")
    stats.probplot(resid, dist="norm", plot=axes[i, 1])
    axes[i, 1].set_title(f"{name}: Normal Q-Q Plot")
plt.tight_layout()
plt.savefig("activity16_residual_diagnostics.png", dpi=150)
plt.show()
