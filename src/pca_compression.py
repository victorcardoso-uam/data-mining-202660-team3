import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================================
# ACTIVITY 5 - DIMENSIONALITY COMPRESSION
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root directory
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        '..'
    )
)

# Portable path to the processed dataset
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'data',
    'processed',
    'solar_features_engineered.csv'
)

# Directory for generated figures
FIGURES_DIR = os.path.join(
    PROJECT_ROOT,
    'reports',
    'figures'
)

# Scree plot output path
SCREE_PLOT_PATH = os.path.join(
    FIGURES_DIR,
    'scree_plot.png'
)

# Compressed PCA dataset output path
COMPRESSED_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'data',
    'processed',
    'solar_compressed_pca.csv'
)

# ============================================================================
# TASK 1: HYPOTHESIS PRE-CHECK & FEATURE ISOLATION
# ============================================================================

print("=" * 70)
print("TASK 1: FEATURE ISOLATION")
print("=" * 70)

print(f"\nScript directory: {SCRIPT_DIR}")
print(f"Dataset path: {DATA_PATH}")

# Verify that the dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# Primary identifiers that must NOT be used in PCA
identifier_columns = [
    'timestamp',
    'panel_id'
]

# Eight continuous variables required by the activity
continuous_features = [
    'voltage_v',
    'current_a',
    'power_w',
    'temperature_c',
    'efficiency_pct',
    'solar_radiation_w_m2',
    'ambient_temp_c',
    'wind_speed_m_s'
]

# Hold identifiers separately
df_identifiers = df[identifier_columns].copy()

# Isolate numerical features for PCA
X = df[continuous_features].copy()

print("\nIdentifier columns held separately:")
print(df_identifiers.dtypes)

print("\nContinuous features isolated for PCA:")
print(X.dtypes)

print(f"\nNumber of continuous features: {X.shape[1]}")
print(f"Number of observations: {X.shape[0]}")

print("\nTASK 1 COMPLETED SUCCESSFULLY")

# ============================================================================
# TASK 2: VARIANCE DISTORTION TRAP & AI SCALING AUDIT
# ============================================================================

print("\n" + "=" * 70)
print("TASK 2: VARIANCE DISTORTION TRAP & AI SCALING AUDIT")
print("=" * 70)

# --------------------------------------------------------------------------
# STEP 1: RAW FEATURE VARIANCES
# --------------------------------------------------------------------------

print("\n--- RAW FEATURE VARIANCES ---")

raw_variances = X.var()

print(raw_variances)

largest_variance_feature = raw_variances.idxmax()

print(
    f"\nFeature with largest raw variance: "
    f"{largest_variance_feature}"
)

print(
    f"Largest raw variance: "
    f"{raw_variances.max():.4f}"
)

# --------------------------------------------------------------------------
# STEP 2: INTENTIONALLY INCORRECT PCA WITHOUT SCALING
# --------------------------------------------------------------------------

print("\n--- UNSCALED PCA (INTENTIONAL AI TRAP) ---")

pca_unscaled = PCA()

X_pca_unscaled = pca_unscaled.fit_transform(X)

unscaled_variance_ratio = pca_unscaled.explained_variance_ratio_

print(
    f"PC1 explained variance ratio: "
    f"{unscaled_variance_ratio[0]:.6f}"
)

print(
    f"PC1 explained variance percentage: "
    f"{unscaled_variance_ratio[0] * 100:.2f}%"
)

# --------------------------------------------------------------------------
# STEP 3: STANDARDIZE CONTINUOUS FEATURES
# --------------------------------------------------------------------------

print("\n--- STANDARD SCALING ---")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=continuous_features
)

print("\nScaled feature means:")
print(X_scaled_df.mean())

print("\nScaled feature variances:")
print(X_scaled_df.var(ddof=0))

# --------------------------------------------------------------------------
# STEP 4: SCALING VERIFICATION
# --------------------------------------------------------------------------

print("\n--- SCALING VERIFICATION ---")

print(
    f"Maximum absolute scaled mean: "
    f"{X_scaled_df.mean().abs().max():.10f}"
)

print(
    f"Minimum scaled variance: "
    f"{X_scaled_df.var(ddof=0).min():.10f}"
)

print(
    f"Maximum scaled variance: "
    f"{X_scaled_df.var(ddof=0).max():.10f}"
)

print("\nTASK 2 COMPLETED SUCCESSFULLY")

# ============================================================================
# TASK 3: UNCONSTRAINED PCA & DIMENSIONALITY TRADE-OFF
# ============================================================================

print("\n" + "=" * 70)
print("TASK 3: UNCONSTRAINED PCA & DIMENSIONALITY TRADE-OFF")
print("=" * 70)

# Fit PCA using all 8 standardized features
pca_full = PCA()

X_pca_full = pca_full.fit_transform(X_scaled)

# Extract individual explained variance ratios
explained_variance_ratio = pca_full.explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# --------------------------------------------------------------------------
# COMPONENT VARIANCE SUMMARY
# --------------------------------------------------------------------------

print("\n--- PCA COMPONENT VARIANCE SUMMARY ---")

variance_summary = pd.DataFrame({
    'Component': [
        f'PC{i + 1}'
        for i in range(len(explained_variance_ratio))
    ],
    'Individual Variance': explained_variance_ratio,
    'Individual Variance (%)': explained_variance_ratio * 100,
    'Cumulative Variance': cumulative_variance,
    'Cumulative Variance (%)': cumulative_variance * 100
})

print(
    variance_summary.to_string(
        index=False,
        formatters={
            'Individual Variance': '{:.6f}'.format,
            'Individual Variance (%)': '{:.2f}'.format,
            'Cumulative Variance': '{:.6f}'.format,
            'Cumulative Variance (%)': '{:.2f}'.format
        }
    )
)

# --------------------------------------------------------------------------
# TWO VS. THREE COMPONENT COMPARISON
# --------------------------------------------------------------------------

two_pc_variance = cumulative_variance[1] * 100
three_pc_variance = cumulative_variance[2] * 100

print("\n--- DIMENSIONALITY TRADE-OFF ---")

print(
    f"Variance retained with 2 principal components: "
    f"{two_pc_variance:.2f}%"
)

print(
    f"Variance retained with 3 principal components: "
    f"{three_pc_variance:.2f}%"
)

variance_gain = three_pc_variance - two_pc_variance

print(
    f"Additional variance retained by using PC3: "
    f"{variance_gain:.2f}%"
)

# Team decision
CHOSEN_COMPONENTS = 3

print(
    f"\nTeam decision: "
    f"{CHOSEN_COMPONENTS} principal components selected."
)

print("\nTASK 3 COMPLETED SUCCESSFULLY")

# ============================================================================
# TASK 4: SCREE PLOT GENERATION & DIMINISHING RETURNS ELBOW
# ============================================================================

print("\n" + "=" * 70)
print("TASK 4: SCREE PLOT GENERATION")
print("=" * 70)

# Create figures directory if it does not exist
os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)

components = np.arange(
    1,
    len(explained_variance_ratio) + 1
)

# Create the Scree Plot
plt.figure(
    figsize=(10, 6)
)

# Individual explained variance bars
plt.bar(
    components,
    explained_variance_ratio,
    alpha=0.7,
    label='Individual Explained Variance'
)

# Cumulative explained variance stepped line
plt.step(
    components,
    cumulative_variance,
    where='mid',
    marker='o',
    linewidth=2,
    label='Cumulative Explained Variance'
)

# 95% variance threshold
plt.axhline(
    y=0.95,
    linestyle='--',
    linewidth=2,
    label='95% Variance Threshold'
)

# Mark selected number of components
plt.axvline(
    x=CHOSEN_COMPONENTS,
    linestyle=':',
    linewidth=2,
    label=f'Selected Components = {CHOSEN_COMPONENTS}'
)

# Titles and labels
plt.title(
    'Scree Plot - PCA Explained Variance'
)

plt.xlabel(
    'Principal Component'
)

plt.ylabel(
    'Explained Variance Ratio'
)

plt.xticks(
    components
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    axis='y',
    alpha=0.3
)

plt.legend()

plt.tight_layout()

# Save figure
plt.savefig(
    SCREE_PLOT_PATH,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("\nScree plot successfully saved to:")
print(SCREE_PLOT_PATH)

print(
    f"\nSelected PCA components: "
    f"{CHOSEN_COMPONENTS}"
)

print(
    f"Cumulative variance at PC{CHOSEN_COMPONENTS}: "
    f"{cumulative_variance[CHOSEN_COMPONENTS - 1] * 100:.2f}%"
)

print("\nTASK 4 COMPLETED SUCCESSFULLY")

# ============================================================================
# TASK 5: COMPRESSED MATRIX RECONSTRUCTION & PRIMARY KEY ALIGNMENT
# ============================================================================

print("\n" + "=" * 70)
print("TASK 5: COMPRESSED MATRIX RECONSTRUCTION")
print("=" * 70)

# Fit PCA using the selected number of components
pca_compressed = PCA(
    n_components=CHOSEN_COMPONENTS
)

X_compressed = pca_compressed.fit_transform(
    X_scaled
)

# Create PC column names
pc_columns = [
    f'PC{i + 1}'
    for i in range(CHOSEN_COMPONENTS)
]

# Reconstruct PCA results as a DataFrame
df_pca = pd.DataFrame(
    X_compressed,
    columns=pc_columns,
    index=df.index
)

# Reset indexes to ensure correct row alignment
df_identifiers_aligned = df_identifiers.reset_index(
    drop=True
)

df_pca = df_pca.reset_index(
    drop=True
)

# Prepend timestamp and panel_id
df_compressed = pd.concat(
    [
        df_identifiers_aligned,
        df_pca
    ],
    axis=1
)

# --------------------------------------------------------------------------
# VERIFY DATAFRAME STRUCTURE
# --------------------------------------------------------------------------

print("\n--- COMPRESSED DATAFRAME HEAD ---")

print(
    df_compressed.head(3)
)

print("\n--- COMPRESSED DATAFRAME SHAPE ---")

print(
    df_compressed.shape
)

print("\n--- COMPRESSED DATAFRAME COLUMNS ---")

print(
    df_compressed.columns.tolist()
)

# --------------------------------------------------------------------------
# PRIMARY KEY ALIGNMENT VERIFICATION
# --------------------------------------------------------------------------

identifier_alignment_valid = (
    df_compressed['timestamp'].equals(
        df['timestamp'].reset_index(drop=True)
    )
    and
    df_compressed['panel_id'].equals(
        df['panel_id'].reset_index(drop=True)
    )
)

print("\n--- PRIMARY KEY ALIGNMENT CHECK ---")

print(
    f"Timestamp and panel_id alignment preserved: "
    f"{identifier_alignment_valid}"
)

# --------------------------------------------------------------------------
# EXPORT COMPRESSED DATASET
# --------------------------------------------------------------------------

df_compressed.to_csv(
    COMPRESSED_DATA_PATH,
    index=False
)

print("\nCompressed dataset successfully exported to:")
print(COMPRESSED_DATA_PATH)

# Verify export
if os.path.exists(COMPRESSED_DATA_PATH):
    print("\nCSV export verification: SUCCESS")
else:
    raise FileNotFoundError(
        "Compressed PCA CSV was not created."
    )

print("\nTASK 5 COMPLETED SUCCESSFULLY")

# ============================================================================
# TASK 6: FACTOR LOADINGS & SEMANTIC LATENT NAMING
# ============================================================================

print("\n" + "=" * 70)
print("TASK 6: FACTOR LOADINGS & SEMANTIC LATENT NAMING")
print("=" * 70)

# --------------------------------------------------------------------------
# EXTRACT FACTOR LOADINGS
# --------------------------------------------------------------------------

# Use loadings from the PCA model fitted on standardized data
loadings = pd.DataFrame(
    pca_compressed.components_.T,
    index=continuous_features,
    columns=pc_columns
)

# Keep PC1 and PC2 for interpretation
loadings_pc1_pc2 = loadings[
    ['PC1', 'PC2']
].copy()

print("\n--- FACTOR LOADINGS: PC1 AND PC2 ---")

print(
    loadings_pc1_pc2.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)

# --------------------------------------------------------------------------
# ABSOLUTE LOADING RANKINGS
# --------------------------------------------------------------------------

print("\n--- TOP ABSOLUTE LOADINGS FOR PC1 ---")

pc1_ranked = (
    loadings_pc1_pc2['PC1']
    .abs()
    .sort_values(
        ascending=False
    )
)

print(pc1_ranked)

print("\n--- TOP ABSOLUTE LOADINGS FOR PC2 ---")

pc2_ranked = (
    loadings_pc1_pc2['PC2']
    .abs()
    .sort_values(
        ascending=False
    )
)

print(pc2_ranked)

# --------------------------------------------------------------------------
# SEMANTIC NAMES
# --------------------------------------------------------------------------

PC1_SEMANTIC_NAME = (
    "Solar Thermal-Electrical Intensity Index"
)

PC2_SEMANTIC_NAME = (
    "Ambient Weather Dynamics Factor"
)

print("\n--- SEMANTIC COMPONENT NAMES ---")

print(
    f"PC1 Semantic Name: "
    f"{PC1_SEMANTIC_NAME}"
)

print(
    f"PC2 Semantic Name: "
    f"{PC2_SEMANTIC_NAME}"
)

print("\nTASK 6 COMPLETED SUCCESSFULLY")