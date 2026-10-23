# ==================================================
# ACTIVITY 16 - MULTI-MODEL REGRESSION
# STEP 2 - DATA INGESTION & 80/20 SPLIT
# ==================================================

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

file_path = "data/raw/regression_team3_ev_battery.csv"

df = pd.read_csv(file_path)


print("\n" + "=" * 65)
print("ACTIVITY 16 - STEP 2: DATA INGESTION")
print("=" * 65)

print(f"\nDataset: {file_path}")
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# --------------------------------------------------
# DATASET STRUCTURE
# --------------------------------------------------

print("\n" + "-" * 65)
print("AVAILABLE COLUMNS")
print("-" * 65)

for column in df.columns:
    print(f" - {column}")


print("\n" + "-" * 65)
print("MISSING VALUES")
print("-" * 65)

print(df.isnull().sum())


# --------------------------------------------------
# DEFINE FEATURES AND TARGET
# --------------------------------------------------

target_column = "battery_cell_temp_c"

feature_columns = [
    "discharge_current_a",
    "ambient_temp_c",
    "state_of_charge_pct",
    "internal_resistance_mohm",
    "coolant_flow_l_min",
    "cell_voltage_v"
]

X = df[feature_columns]
y = df[target_column]


print("\n" + "=" * 65)
print("FEATURES AND TARGET")
print("=" * 65)

print(f"\nTarget:")
print(f" - {target_column}")

print("\nPredictor pool:")
for feature in feature_columns:
    print(f" - {feature}")


# --------------------------------------------------
# 80/20 TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n" + "=" * 65)
print("TRAIN / TEST SPLIT")
print("=" * 65)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

print(f"Training share: {len(X_train) / len(df):.1%}")
print(f"Testing share:  {len(X_test) / len(df):.1%}")

print("\nRandom state: 42")
print("The same split will be used for all 5 regression models.")


print("\n" + "=" * 65)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 65 + "\n")
