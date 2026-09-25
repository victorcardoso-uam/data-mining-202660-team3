import os
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================================
# ACTIVITY 7 - UNSUPERVISED MARKET SEGMENTATION & RFM CLUSTERING
# TEAM 3 - INDUSTRIAL MRO
# TASKS 1, 2, 3 & 4
# ============================================================================

# Resolve project paths dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "segmentation_team3_mro.csv"
)

print("=" * 70)
print("ACTIVITY 7 - RFM MARKET SEGMENTATION")
print("TEAM 3 - INDUSTRIAL MRO")
print("=" * 70)

print(f"\nScript directory: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Dataset path: {DATA_PATH}")

# ============================================================================
# LOAD DATASET
# ============================================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# ============================================================================
# IDENTIFY RFM FEATURES
# ============================================================================

ID_COLUMN = "factory_id"

RFM_COLUMNS = [
    "recency_replenishment_days",
    "frequency_mro_orders",
    "mro_annual_spend_mxn"
]

METADATA_COLUMN = "avg_lead_time_days"

print("\nRFM features identified:")
print(f"Recency   : {RFM_COLUMNS[0]}")
print(f"Frequency : {RFM_COLUMNS[1]}")
print(f"Monetary  : {RFM_COLUMNS[2]}")

print(f"\nPrimary ID column: {ID_COLUMN}")
print(f"Secondary metadata column: {METADATA_COLUMN}")

# ============================================================================
# TASK 1 - RAW FEATURE SKEWNESS
# ============================================================================

rfm_raw = df[RFM_COLUMNS].copy()
raw_skewness = rfm_raw.skew()

print("\n" + "=" * 70)
print("TASK 1 - RAW RFM SKEWNESS COEFFICIENTS")
print("=" * 70)

print(
    f"Recency Skewness  : "
    f"{raw_skewness['recency_replenishment_days']:.6f}"
)

print(
    f"Frequency Skewness: "
    f"{raw_skewness['frequency_mro_orders']:.6f}"
)

print(
    f"Monetary Skewness : "
    f"{raw_skewness['mro_annual_spend_mxn']:.6f}"
)

# ============================================================================
# TASK 2 - RAW VARIANCE
# ============================================================================

raw_variance = rfm_raw.var()

print("\n" + "=" * 70)
print("TASK 2 - RAW RFM VARIANCES")
print("=" * 70)

print(
    f"Recency Raw Variance  : "
    f"{raw_variance['recency_replenishment_days']:.6f}"
)

print(
    f"Frequency Raw Variance: "
    f"{raw_variance['frequency_mro_orders']:.6f}"
)

print(
    f"Monetary Raw Variance : "
    f"{raw_variance['mro_annual_spend_mxn']:.6f}"
)

# ============================================================================
# TASK 2 - LOG1P VARIANCE STABILIZATION
# ============================================================================

rfm_log = np.log1p(rfm_raw)

log_variance = rfm_log.var()
log_skewness = rfm_log.skew()

print("\n" + "=" * 70)
print("TASK 2 - LOG-TRANSFORMED RFM VARIANCES")
print("=" * 70)

print(
    f"Log-Recency Variance  : "
    f"{log_variance['recency_replenishment_days']:.6f}"
)

print(
    f"Log-Frequency Variance: "
    f"{log_variance['frequency_mro_orders']:.6f}"
)

print(
    f"Log-Monetary Variance : "
    f"{log_variance['mro_annual_spend_mxn']:.6f}"
)

print("\n" + "=" * 70)
print("TASK 2 - LOG-TRANSFORMED RFM SKEWNESS")
print("=" * 70)

print(
    f"Log-Recency Skewness  : "
    f"{log_skewness['recency_replenishment_days']:.6f}"
)

print(
    f"Log-Frequency Skewness: "
    f"{log_skewness['frequency_mro_orders']:.6f}"
)

print(
    f"Log-Monetary Skewness : "
    f"{log_skewness['mro_annual_spend_mxn']:.6f}"
)

# ============================================================================
# TASK 3 - STANDARDIZE LOG-TRANSFORMED FEATURES
# ============================================================================

scaler = StandardScaler()

rfm_scaled_array = scaler.fit_transform(rfm_log)

rfm_scaled = pd.DataFrame(
    rfm_scaled_array,
    columns=RFM_COLUMNS,
    index=df.index
)

scaled_means = np.mean(rfm_scaled_array, axis=0)
scaled_variances = np.var(rfm_scaled_array, axis=0, ddof=0)

print("\n" + "=" * 70)
print("TASK 3 - STANDARDIZATION VERIFICATION")
print("=" * 70)

print("\nScaled feature means (expected approximately 0.0):")
for column, mean_value in zip(RFM_COLUMNS, scaled_means):
    print(f"{column}: {mean_value:.10f}")

print("\nScaled feature variances (expected approximately 1.0):")
for column, variance_value in zip(RFM_COLUMNS, scaled_variances):
    print(f"{column}: {variance_value:.10f}")

print("\nFirst 3 standardized rows:")
print(rfm_scaled.head(3))

print(f"\nScaled feature matrix shape: {rfm_scaled.shape}")

# ============================================================================
# TASK 4 - K-MEANS TUNING FROM K=2 TO K=10
# ============================================================================

K_VALUES = range(2, 11)

inertia_scores = []
silhouette_scores = []

print("\n" + "=" * 70)
print("TASK 4 - K-MEANS TUNING SCORES")
print("=" * 70)

print("\nK | Inertia | Silhouette")
print("-" * 42)

for k in K_VALUES:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(rfm_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(
        rfm_scaled,
        cluster_labels
    )

    inertia_scores.append(inertia)
    silhouette_scores.append(silhouette)

    print(
        f"K={k:2d} | "
        f"Inertia = {inertia:.6f} | "
        f"Silhouette = {silhouette:.6f}"
    )

# ============================================================================
# TASK 4 - NUMERICAL SUMMARY
# ============================================================================

best_silhouette_index = int(np.argmax(silhouette_scores))
best_silhouette_k = list(K_VALUES)[best_silhouette_index]
best_silhouette_value = silhouette_scores[best_silhouette_index]

print("\n" + "=" * 70)
print("TASK 4 - NUMERICAL TUNING SUMMARY")
print("=" * 70)

print(
    f"Highest Silhouette Score: "
    f"{best_silhouette_value:.6f}"
)

print(
    f"K with Highest Silhouette: "
    f"{best_silhouette_k}"
)

print(
    "\nNOTE: The final optimal K has NOT been selected yet. "
    "The Inertia trade-off must also be evaluated."
)

print("\nTask 4 K-Means tuning completed successfully.")
# ============================================================================
# TASK 5 - DUAL-PANEL DIAGNOSTIC VISUALIZATIONS
# ============================================================================

OPTIMAL_K = 3

FIGURES_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "figures"
)

FIGURE_PATH = os.path.join(
    FIGURES_DIR,
    "rfm_diagnostics.png"
)

# Create directory if it does not already exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# Convert K range to a list for plotting
k_values_list = list(K_VALUES)

# Create side-by-side diagnostic plots
fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

# --------------------------------------------------------------------------
# LEFT PANEL - INERTIA ELBOW CURVE
# --------------------------------------------------------------------------

axes[0].plot(
    k_values_list,
    inertia_scores,
    marker="o",
    label="Inertia (WCSS)"
)

axes[0].axvline(
    x=OPTIMAL_K,
    linestyle="--",
    label=f"Selected K = {OPTIMAL_K}"
)

axes[0].set_title("K-Means Inertia Elbow Curve")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_xticks(k_values_list)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# --------------------------------------------------------------------------
# RIGHT PANEL - SILHOUETTE CURVE
# --------------------------------------------------------------------------

axes[1].plot(
    k_values_list,
    silhouette_scores,
    marker="o",
    label="Silhouette Score"
)

axes[1].axvline(
    x=OPTIMAL_K,
    linestyle="--",
    label=f"Selected K = {OPTIMAL_K}"
)

axes[1].set_title("K-Means Silhouette Coefficient")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(k_values_list)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# --------------------------------------------------------------------------
# SAVE FIGURE
# --------------------------------------------------------------------------

fig.suptitle(
    "Team 3 Industrial MRO - RFM K-Means Diagnostics",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\n" + "=" * 70)
print("TASK 5 - DIAGNOSTIC VISUALIZATION")
print("=" * 70)

print(f"Selected optimal K: {OPTIMAL_K}")
print(f"Diagnostic figure saved to: {FIGURE_PATH}")

print("\nTask 5 diagnostic visualization completed successfully.")
# ============================================================================
# TASK 6 - FINAL OPTIMAL K FITTING & CLUSTER CENTROID EXTRACTION
# ============================================================================

FINAL_K = 3

# Fit the final K-Means model using the selected optimal K
final_kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=10
)

final_cluster_labels = final_kmeans.fit_predict(rfm_scaled)

# ============================================================================
# ATTACH CLUSTER ID TO ORIGINAL DATAFRAME
# ============================================================================

df_clustered = df.copy()
df_clustered["cluster_id"] = final_cluster_labels

print("\n" + "=" * 70)
print("TASK 6 - FINAL K-MEANS CLUSTER ASSIGNMENTS")
print("=" * 70)

print(f"\nFinal K selected: {FINAL_K}")
print(f"Original dataset shape: {df.shape}")
print(f"Clustered dataset shape: {df_clustered.shape}")

print("\nFirst 10 rows with original ID and cluster assignment:")
print(
    df_clustered[
        [
            ID_COLUMN,
            "recency_replenishment_days",
            "frequency_mro_orders",
            "mro_annual_spend_mxn",
            "cluster_id"
        ]
    ].head(10)
)

# ============================================================================
# VERIFY ORIGINAL IDS REMAIN INTACT
# ============================================================================

ids_preserved = df_clustered[ID_COLUMN].equals(df[ID_COLUMN])

print("\nOriginal ID preservation check:")
print(f"factory_id preserved correctly: {ids_preserved}")

# ============================================================================
# CLUSTER SIZES
# ============================================================================

cluster_sizes = (
    df_clustered["cluster_id"]
    .value_counts()
    .sort_index()
)

print("\nCluster sizes:")
for cluster_id, size in cluster_sizes.items():
    print(f"Cluster {cluster_id}: {size} factories")

# ============================================================================
# CENTROIDS IN ORIGINAL BUSINESS UNITS
# ============================================================================

original_centroids = (
    df_clustered
    .groupby("cluster_id")[RFM_COLUMNS]
    .mean()
)

print("\n" + "=" * 70)
print("TASK 6 - CENTROIDS IN ORIGINAL BUSINESS UNITS")
print("=" * 70)

for cluster_id, row in original_centroids.iterrows():
    print(f"\nCluster {cluster_id}:")
    print(
        f"  Recency (days)       : "
        f"{row['recency_replenishment_days']:.2f}"
    )
    print(
        f"  Frequency (orders)   : "
        f"{row['frequency_mro_orders']:.2f}"
    )
    print(
        f"  Monetary (MXN)       : "
        f"${row['mro_annual_spend_mxn']:,.2f}"
    )
    print(
        f"  Cluster Size         : "
        f"{cluster_sizes.loc[cluster_id]} factories"
    )

print("\nTask 6 final clustering and centroid extraction completed successfully.")
# ============================================================================
# TASK 7 - SECONDARY METADATA VALIDATION FOR BUSINESS PERSONAS
# ============================================================================

metadata_by_cluster = (
    df_clustered
    .groupby("cluster_id")[METADATA_COLUMN]
    .agg(["mean", "median", "min", "max"])
)

print("\n" + "=" * 70)
print("TASK 7 - SECONDARY METADATA BY CLUSTER")
print("=" * 70)

for cluster_id, row in metadata_by_cluster.iterrows():
    print(f"\nCluster {cluster_id}:")
    print(f"  Average Lead Time : {row['mean']:.2f} days")
    print(f"  Median Lead Time  : {row['median']:.2f} days")
    print(f"  Minimum Lead Time : {row['min']:.2f} days")
    print(f"  Maximum Lead Time : {row['max']:.2f} days")

print("\n" + "=" * 70)
print("TASK 7 - PERSONA EVIDENCE SUMMARY")
print("=" * 70)

for cluster_id in sorted(df_clustered["cluster_id"].unique()):
    cluster_data = df_clustered[
        df_clustered["cluster_id"] == cluster_id
    ]

    print(f"\nCluster {cluster_id}:")
    print(
        f"  Factories              : "
        f"{len(cluster_data)}"
    )
    print(
        f"  Avg Recency            : "
        f"{cluster_data['recency_replenishment_days'].mean():.2f} days"
    )
    print(
        f"  Avg Frequency          : "
        f"{cluster_data['frequency_mro_orders'].mean():.2f} orders"
    )
    print(
        f"  Avg Monetary           : "
        f"${cluster_data['mro_annual_spend_mxn'].mean():,.2f} MXN"
    )
    print(
        f"  Avg Lead Time          : "
        f"{cluster_data['avg_lead_time_days'].mean():.2f} days"
    )

print("\nTask 7 metadata validation completed successfully.")