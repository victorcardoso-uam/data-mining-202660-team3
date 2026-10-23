import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ============================================================================
# ACTIVITY 13 - DECISION TREE COST-COMPLEXITY PRUNING
# TEAM 3 - EV BATTERY THERMAL RUNAWAY (UrbanMart / Mobility)
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "pruning_team3_ev_battery.csv")
TARGET = "thermal_runaway_risk"

# Step 2: Data ingestion & stratified 75/25 split
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Class balance:\n{df[TARGET].value_counts()}\n")

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

# Step 3: Multi-alpha experimentation
candidates = {
    "alpha_1 (Unconstrained)": 0.001,
    "alpha_2 (Optimal Pruned)": 0.015,
    "alpha_3 (Over-Pruned)": 0.080,
}

results = []
for name, alpha in candidates.items():
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)

    # Step 4: Complexity & error metrics
    leaves = clf.get_n_leaves()
    depth = clf.get_depth()
    r_train = 1.0 - clf.score(X_train, y_train)
    r_test = 1.0 - clf.score(X_test, y_test)

    # Step 5: Total cost R_alpha(T) = R_test(T) + alpha * |T|
    total_cost = r_test + alpha * leaves

    results.append({
        "Candidate": name,
        "ccp_alpha": alpha,
        "Leaves |T|": leaves,
        "Max Depth": depth,
        "Train Error %": round(r_train * 100, 2),
        "Test Error %": round(r_test * 100, 2),
        "R_alpha(T)": round(total_cost, 4),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best = results_df.loc[results_df["R_alpha(T)"].idxmin()]
print(f"\nOptimal alpha* = {best['ccp_alpha']} ({best['Candidate']})")
print(f"Min total cost R_alpha(T) = {best['R_alpha(T)']} | Test error = {best['Test Error %']}%")
# Activity 13 - Decision Tree Pruning
# Team 3 - EV Battery Thermal Runaway

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ==================================================
# STEP 2 - DATA INGESTION & STRATIFIED SPLIT
# ==================================================

print("\n" + "=" * 60)
print("ACTIVITY 13 - TEAM 3")
print("EV BATTERY THERMAL RUNAWAY")
print("=" * 60)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/processed/pruning_team3_ev_battery.csv")

print("\n[1] DATASET LOADED SUCCESSFULLY")
print("-" * 60)
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# --------------------------------------------------
# 2. DISPLAY COLUMNS
# --------------------------------------------------

print("\n[2] DATASET COLUMNS")
print("-" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# --------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

target = "thermal_runaway_risk"

X = df.drop(columns=[target])
y = df[target]

print("\n[3] FEATURES AND TARGET")
print("-" * 60)
print(f"Target variable: {target}")

print("\nPredictor variables:")
for column in X.columns:
    print(f" - {column}")


# --------------------------------------------------
# 4. STRATIFIED TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\n[4] TRAIN / TEST SPLIT")
print("-" * 60)
print(f"Training samples: {len(X_train)} ({len(X_train) / len(df):.1%})")
print(f"Testing samples:  {len(X_test)} ({len(X_test) / len(df):.1%})")


# --------------------------------------------------
# 5. VERIFY STRATIFICATION
# --------------------------------------------------

print("\n[5] TARGET DISTRIBUTION")
print("-" * 60)

train_distribution = y_train.value_counts(normalize=True) * 100
test_distribution = y_test.value_counts(normalize=True) * 100

print("\nTRAINING SET:")
for value, percentage in train_distribution.items():
    print(f" Class {value}: {percentage:.2f}%")

print("\nTEST SET:")
for value, percentage in test_distribution.items():
    print(f" Class {value}: {percentage:.2f}%")


# --------------------------------------------------
# FINAL CONFIRMATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("75% Training | 25% Testing")
print("random_state = 42 | stratify = y")
print("=" * 60 + "\n")

# ==================================================
# STEP 3 - MULTI-ALPHA PARAMETER EXPERIMENTATION
# ==================================================

print("\n" + "=" * 60)
print("STEP 3 - DECISION TREE PRUNING EXPERIMENT")
print("=" * 60)


alpha_candidates = {
    "Alpha 1 - Unconstrained": 0.001,
    "Alpha 2 - Optimal Pruned": 0.015,
    "Alpha 3 - Over-Pruned": 0.080
}


models = {}


for model_name, alpha in alpha_candidates.items():

    model = DecisionTreeClassifier(
        ccp_alpha=alpha,
        random_state=42
    )

    model.fit(X_train, y_train)

    models[model_name] = model

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)
    print(f"ccp_alpha: {alpha}")
    print("Model trained successfully")


print("\n" + "=" * 60)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("3 DECISION TREE MODELS TRAINED")
print("=" * 60 + "\n")

# ==================================================
# STEP 4 - MODEL COMPLEXITY & ERROR METRICS
# ==================================================

print("\n" + "=" * 70)
print("STEP 4 - MODEL COMPLEXITY & ERROR METRICS")
print("=" * 70)

results = []

for model_name, model in models.items():

    alpha = model.ccp_alpha

    leaf_count = model.get_n_leaves()
    max_depth = model.get_depth()

    train_error = 1.0 - model.score(X_train, y_train)
    test_error = 1.0 - model.score(X_test, y_test)

    results.append({
        "Model": model_name,
        "Alpha": alpha,
        "Leaf Count": leaf_count,
        "Max Depth": max_depth,
        "Train Error": train_error,
        "Test Error": test_error
    })

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)
    print(f"ccp_alpha:   {alpha}")
    print(f"Leaf Count:  {leaf_count}")
    print(f"Max Depth:   {max_depth}")
    print(f"Train Error: {train_error:.2%}")
    print(f"Test Error:  {test_error:.2%}")


print("\n" + "=" * 70)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 70 + "\n")

# ==================================================
# STEP 5 - TOTAL COST & OPTIMAL ALPHA SELECTION
# ==================================================

print("\n" + "=" * 70)
print("STEP 5 - TOTAL COST-COMPLEXITY ANALYSIS")
print("=" * 70)

for result in results:
    total_cost = (
        result["Test Error"]
        + result["Alpha"] * result["Leaf Count"]
    )

    result["Total Cost"] = total_cost

    print("\n" + "-" * 70)
    print(result["Model"])
    print("-" * 70)
    print(f"Alpha:       {result['Alpha']}")
    print(f"Test Error:  {result['Test Error']:.2%}")
    print(f"Leaf Count:  {result['Leaf Count']}")
    print(f"Total Cost:  {total_cost:.4f}")


# --------------------------------------------------
# SELECT OPTIMAL MODEL
# --------------------------------------------------

best_model = min(
    results,
    key=lambda x: x["Total Cost"]
)

print("\n" + "=" * 70)
print("OPTIMAL MODEL")
print("=" * 70)

print(f"Model:       {best_model['Model']}")
print(f"Alpha*:      {best_model['Alpha']}")
print(f"Leaf Count:  {best_model['Leaf Count']}")
print(f"Max Depth:   {best_model['Max Depth']}")
print(f"Train Error: {best_model['Train Error']:.2%}")
print(f"Test Error:  {best_model['Test Error']:.2%}")
print(f"Total Cost:  {best_model['Total Cost']:.4f}")

print("\n" + "=" * 70)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 70 + "\n")