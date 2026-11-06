import os
import pandas as pd
import numpy as np


# =============================================================================
# STEP 1: PORTABILITY INITIALIZATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(
    SCRIPT_DIR,
    "..",
    "data",
    "raw",
    "solar_telemetry_corrupted.csv"
)

print(f"Resolved Path: {RAW_DATA_PATH}")

df = pd.read_csv(RAW_DATA_PATH)

print(f"Initial Shape: {df.shape}")
print(df.head(3))


# =============================================================================
# STEP 2: NUMERICAL IMPUTATION
# =============================================================================

numerical_features = [
    "voltage_v",
    "current_a",
    "power_w",
    "efficiency_pct",
    "temperature_c"
]

print("\n--- NUMERICAL IMPUTATION ---")

for col in numerical_features:

    median_value = df[col].median()

    print(
        f"Calculated median for {col:15}: "
        f"{median_value:.4f}"
    )

    df[col] = df[col].fillna(median_value)


print("\nNulls remaining in numerical columns:")
print(df[numerical_features].isnull().sum())


# =============================================================================
# STEP 3: CATEGORICAL STANDARDIZATION
# =============================================================================

categorical_features = [
    "panel_id",
    "timestamp"
]

print("\n--- CATEGORICAL STANDARDIZATION ---")

for col in categorical_features:
    df[col] = df[col].fillna("Unknown")


print("Nulls remaining in categorical columns:")
print(df[categorical_features].isnull().sum())


# =============================================================================
# STEP 4: PHYSICAL OUTLIERS & TEMPERATURE WINSORIZATION
# =============================================================================

print("\n--- OUTLIER HANDLING ---")


# Efficiency outliers
efficiency_outliers_below = df[
    df["efficiency_pct"] < 0.0
].index.tolist()

efficiency_outliers_above = df[
    df["efficiency_pct"] > 100.0
].index.tolist()


print(
    "Efficiency below 0.0% row indexes:",
    efficiency_outliers_below
)

print(
    "Efficiency above 100.0% row indexes:",
    efficiency_outliers_above
)


# Clip efficiency between 0 and 100
df["efficiency_pct"] = df["efficiency_pct"].clip(
    lower=0.0,
    upper=100.0
)


# Calculate 95th percentile of temperature
p95_temperature = df["temperature_c"].quantile(0.95)

print(
    f"Robust 95th Percentile Temperature: "
    f"{p95_temperature:.4f}"
)


# Count temperatures greater than 45 C
temperature_outliers_count = (
    df["temperature_c"] > 45.0
).sum()

print(
    "Count of temperature rows above 45 C:",
    temperature_outliers_count
)


# Cap temperature outliers
df.loc[
    df["temperature_c"] > 45.0,
    "temperature_c"
] = p95_temperature


# =============================================================================
# STEP 5: EXPORT CLEANED DATASET
# =============================================================================

PROCESSED_DIR = os.path.join(
    SCRIPT_DIR,
    "..",
    "data",
    "processed"
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

CLEANED_DATASET_PATH = os.path.join(
    PROCESSED_DIR,
    "solar_telemetry_cleaned.csv"
)


df.to_csv(
    CLEANED_DATASET_PATH,
    index=False
)


# =============================================================================
# FINAL VERIFICATION
# =============================================================================

print("\n--- FINAL VERIFICATION ---")

print(
    "Final cleaned dataset shape:",
    df.shape
)

print("\nFinal missing value counts:")
print(df.isnull().sum())

print(
    "\nCleaned dataset exported to:",
    CLEANED_DATASET_PATH
)