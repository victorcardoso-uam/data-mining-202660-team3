from pathlib import Path

import pandas as pd


# Locate the dataset.
PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data" / "regression_team3_ev_battery.csv"

print("\nACTIVITY 16 - TEAM 3")

print("\nDataset path:")
print(CSV_PATH)

# Check that the CSV exists.
if not CSV_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}"
    )

# Load the CSV into a DataFrame.
df = pd.read_csv(CSV_PATH)

# Remove spaces around column names.
df.columns = df.columns.str.strip()

# Display the first five rows.
print("\nFirst five rows:")
print(df.head())

# Display the dataset size.
print("\nDataset dimensions:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Display column names and data types.
print("\nColumn names:")
print(df.columns.tolist())

print("\nColumn information:")
df.info()

# Check missing values and duplicate rows.
print("\nMissing values per column:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nCSV loaded and inspected successfully.")

from sklearn.model_selection import train_test_split


# Define the target variable.
target = "battery_cell_temp_c"

# Define the available predictors.
predictors = [
    "discharge_current_a",
    "ambient_temp_c",
    "state_of_charge_pct",
    "internal_resistance_mohm",
    "coolant_flow_l_min",
    "cell_voltage_v",
]

# Separate predictors and target.
X = df[predictors].copy()
y = df[target].copy()

# Create one shared split for all five models.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# Display the split sizes.
print("\nTRAIN / TEST SPLIT")

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

print("\nPredictor dimensions:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

print("\nTarget dimensions:")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# Verify that predictors and target remain aligned.
assert X_train.index.equals(y_train.index)
assert X_test.index.equals(y_test.index)

# Verify that training and testing records do not overlap.
assert set(X_train.index).isdisjoint(X_test.index)

print("\nSplit checks passed.")

# Define the predictor combinations for the five models.
model_features = {
    "Model 1": [
        "discharge_current_a",
    ],
    "Model 2": [
        "discharge_current_a",
        "internal_resistance_mohm",
    ],
    "Model 3": [
        "discharge_current_a",
        "internal_resistance_mohm",
        "ambient_temp_c",
    ],
    "Model 4": [
        "discharge_current_a",
        "internal_resistance_mohm",
        "ambient_temp_c",
        "coolant_flow_l_min",
    ],
    "Model 5": [
        "discharge_current_a",
        "internal_resistance_mohm",
        "ambient_temp_c",
        "coolant_flow_l_min",
        "state_of_charge_pct",
    ],
}

# Check and display each model's predictors.
print("\nMODEL CONFIGURATIONS")

for expected_count, (model_name, features) in enumerate(
    model_features.items(),
    start=1,
):
    assert len(features) == expected_count
    assert len(features) == len(set(features))
    assert all(feature in X_train.columns for feature in features)

    print(f"\n{model_name}")
    print(f"Number of predictors: {len(features)}")
    print(f"Selected features: {', '.join(features)}")

print("\nAll five model configurations passed.")

from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# Store metrics, fitted models, and predictions.
results = []
trained_models = {}
test_predictions = {}

# Train and evaluate each model using the same split.
for model_name, features in model_features.items():

    # Select this model's predictors.
    X_train_selected = X_train[features]
    X_test_selected = X_test[features]

    # Fit the model using training data only.
    model = LinearRegression()
    model.fit(X_train_selected, y_train)

    # Generate training and testing predictions.
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)

    # Calculate R-squared.
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Calculate adjusted R-squared on the training set.
    n_train = len(y_train)
    p = len(features)

    if n_train <= p + 1:
        raise ValueError(
            "Not enough training records for adjusted R-squared."
        )

    adjusted_r2_train = 1 - (
        (1 - train_r2) * (n_train - 1) / (n_train - p - 1)
    )

    # Calculate prediction errors on the test set.
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))

    # Store results.
    results.append({
        "Model": model_name,
        "Features": ", ".join(features),
        "Predictors": p,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Adjusted R2 (Train)": adjusted_r2_train,
        "Test MAE (C)": test_mae,
        "Test RMSE (C)": test_rmse,
    })

    trained_models[model_name] = model
    test_predictions[model_name] = y_test_pred

# Build the comparison table.
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")

metric_columns = [
    "Model",
    "Predictors",
    "Train R2",
    "Test R2",
    "Adjusted R2 (Train)",
    "Test MAE (C)",
    "Test RMSE (C)",
]

print(
    results_df[metric_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

# Save the complete table, including feature names.
REPORT_DIR = PROJECT_DIR / "reports" / "activity_16"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = REPORT_DIR / "model_comparison.csv"
results_df.to_csv(RESULTS_PATH, index=False)

print("\nResults saved to:")
print(RESULTS_PATH)

print("\nAll five models were trained and evaluated.")

import matplotlib.pyplot as plt
from scipy import stats


# Select the two candidates with the lowest test RMSE.
top_models = (
    results_df
    .sort_values("Test RMSE (C)")
    .head(2)["Model"]
    .tolist()
)

print("\nRESIDUAL DIAGNOSTICS")
print(f"Selected candidates: {', '.join(top_models)}")

# Use a shared residual scale for easier comparison.
residual_limit = max(
    abs(y_test - test_predictions[model_name]).max()
    for model_name in top_models
) * 1.10

prediction_min = min(
    test_predictions[model_name].min()
    for model_name in top_models
)

prediction_max = max(
    test_predictions[model_name].max()
    for model_name in top_models
)

prediction_margin = max(
    (prediction_max - prediction_min) * 0.05,
    0.1,
)

for model_name in top_models:
    predictions = test_predictions[model_name]
    residuals = y_test - predictions

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    # Panel A: predicted values versus test residuals.
    axes[0].scatter(
        predictions,
        residuals,
        alpha=0.70,
        edgecolors="black",
        linewidths=0.4,
    )

    axes[0].axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=1.5,
    )

    axes[0].set_title("Test Residuals vs. Predicted Values")
    axes[0].set_xlabel("Predicted Cell Temperature (C)")
    axes[0].set_ylabel("Residual: Actual - Predicted (C)")

    axes[0].set_ylim(
        -residual_limit,
        residual_limit,
    )

    axes[0].set_xlim(
        prediction_min - prediction_margin,
        prediction_max + prediction_margin,
    )

    axes[0].grid(alpha=0.25)

    # Panel B: normal probability plot.
    stats.probplot(
        residuals.to_numpy(),
        dist="norm",
        plot=axes[1],
    )

    axes[1].set_title("Normal Probability Plot of Test Residuals")
    axes[1].set_xlabel("Theoretical Normal Quantiles")
    axes[1].set_ylabel("Ordered Residuals (C)")
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        f"{model_name} - Residual Diagnostics",
        fontsize=14,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    file_name = (
        model_name.lower().replace(" ", "_")
        + "_residual_diagnostics.png"
    )

    figure_path = REPORT_DIR / file_name

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\n{model_name}")
    print(f"Mean test residual (C): {residuals.mean():.4f}")
    print(f"Diagnostic plot saved to: {figure_path}")

print("\nResidual diagnostic plots saved successfully.")