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