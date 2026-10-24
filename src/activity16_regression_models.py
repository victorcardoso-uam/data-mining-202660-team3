"""Activity 16: comparison of five linear-regression models for EV batteries."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "regression_team3_ev_battery.csv"
TARGET = "battery_cell_temp_c"

# The order reflects physical relevance and the observed linear relationship
# with battery-cell temperature. The sixth available predictor, cell_voltage_v,
# is excluded because it has the weakest absolute correlation with the target.
MODEL_FEATURES = {
    "Model 1": ["ambient_temp_c"],
    "Model 2": ["ambient_temp_c", "discharge_current_a"],
    "Model 3": ["ambient_temp_c", "discharge_current_a", "coolant_flow_l_min"],
    "Model 4": [
        "ambient_temp_c",
        "discharge_current_a",
        "coolant_flow_l_min",
        "internal_resistance_mohm",
    ],
    "Model 5": [
        "ambient_temp_c",
        "discharge_current_a",
        "coolant_flow_l_min",
        "internal_resistance_mohm",
        "state_of_charge_pct",
    ],
}


def adjusted_r2(r2: float, observations: int, predictors: int) -> float:
    """Return adjusted R-squared for the test sample."""
    return 1 - (1 - r2) * (observations - 1) / (observations - predictors - 1)


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    required = {TARGET, *(feature for features in MODEL_FEATURES.values() for feature in features)}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if data[list(required)].isna().any().any():
        raise ValueError("The selected modeling columns contain missing values.")

    all_features = sorted(required.difference({TARGET}))
    X_train, X_test, y_train, y_test = train_test_split(
        data[all_features], data[TARGET], test_size=0.20, random_state=42
    )

    results = []
    predictions = {}
    for model_name, features in MODEL_FEATURES.items():
        model = LinearRegression()
        model.fit(X_train[features], y_train)
        train_prediction = model.predict(X_train[features])
        test_prediction = model.predict(X_test[features])
        test_r2 = r2_score(y_test, test_prediction)

        results.append(
            {
                "Model": model_name,
                "Features": ", ".join(features),
                "Train R2": r2_score(y_train, train_prediction),
                "Test R2": test_r2,
                "Adjusted R2": adjusted_r2(test_r2, len(y_test), len(features)),
                "MAE": mean_absolute_error(y_test, test_prediction),
                "RMSE": np.sqrt(mean_squared_error(y_test, test_prediction)),
            }
        )
        predictions[model_name] = test_prediction

    results_df = pd.DataFrame(results).sort_values("Model")
    results_df.to_csv(BASE_DIR / "activity16_model_comparison.csv", index=False)
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    top_models = results_df.nlargest(2, "Adjusted R2")["Model"].tolist()
    figure, axes = plt.subplots(len(top_models), 2, figsize=(12, 8), constrained_layout=True)
    for row, model_name in enumerate(top_models):
        fitted = predictions[model_name]
        residuals = y_test.to_numpy() - fitted

        axes[row, 0].scatter(fitted, residuals, alpha=0.72, edgecolor="none")
        axes[row, 0].axhline(0, color="crimson", linestyle="--", linewidth=1.2)
        axes[row, 0].set_title(f"{model_name}: fitted values vs residuals")
        axes[row, 0].set_xlabel("Predicted battery cell temperature (°C)")
        axes[row, 0].set_ylabel("Residual (°C)")
        axes[row, 0].grid(alpha=0.2)

        stats.probplot(residuals, dist="norm", plot=axes[row, 1])
        axes[row, 1].set_title(f"{model_name}: normal Q-Q plot")
        axes[row, 1].grid(alpha=0.2)

    figure.suptitle("Residual diagnostics for the two strongest regression models", fontsize=14)
    figure.savefig(BASE_DIR / "activity16_residual_diagnostics.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
