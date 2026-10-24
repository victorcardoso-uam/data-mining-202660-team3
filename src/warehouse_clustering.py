import os
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================================
# ACTIVITY 6 - TEAM 3
# UNSUPERVISED K-MEANS CLUSTERING
# TASK 2 + TASK 3 + TASK 4 + TASK 5
# ============================================================================


# ----------------------------------------------------------------------------
# STEP 1: PORTABLE PATH RESOLUTION
# ----------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        '..'
    )
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    'data',
    'raw',
    'warehouse_sku_team3.csv'
)

FIGURES_DIR = os.path.join(
    PROJECT_ROOT,
    'reports',
    'figures'
)

FIGURE_PATH = os.path.join(
    FIGURES_DIR,
    'warehouse_diagnostics.png'
)

print('=' * 70)
print('ACTIVITY 6 - WAREHOUSE CLUSTERING')
print('TEAM 3')
print('=' * 70)

print(f'Script directory: {SCRIPT_DIR}')
print(f'Project root: {PROJECT_ROOT}')
print(f'Dataset path: {DATASET_PATH}')


# ----------------------------------------------------------------------------
# STEP 2: LOAD DATASET
# ----------------------------------------------------------------------------

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f'Dataset not found at: {DATASET_PATH}'
    )

df = pd.read_csv(DATASET_PATH)

print('\nDataset loaded successfully.')
print(f'Dataset shape: {df.shape}')


# ----------------------------------------------------------------------------
# STEP 3: SELECT TEAM 3 CONTINUOUS FEATURES
# ----------------------------------------------------------------------------

features = [
    'daily_picking_frequency',
    'unit_weight_kg',
    'humidity_tolerance_pct',
    'shelf_life_days'
]

missing_columns = [
    col for col in features
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f'The following required columns are missing: {missing_columns}'
    )

X = df[features].copy()

print('\nContinuous numerical features selected:')
print(features)

print('\nFeature data types:')
print(X.dtypes)


# ============================================================================
# TASK 2: DISTANCE DISTORTION TRAP & AI SCALING AUDIT
# ============================================================================

print('\n' + '=' * 70)
print('TASK 2: DISTANCE DISTORTION & SCALING AUDIT')
print('=' * 70)


# ----------------------------------------------------------------------------
# STEP 4: RAW FEATURE VARIANCES
# ----------------------------------------------------------------------------

raw_variances = X.var()

print('\nRAW FEATURE VARIANCES')
print('-' * 70)
print(raw_variances)

largest_variance_feature = raw_variances.idxmax()

print(
    f'\nFeature with the largest raw variance: '
    f'{largest_variance_feature}'
)

print(
    f'Largest raw variance value: '
    f'{raw_variances.max():.6f}'
)


# ----------------------------------------------------------------------------
# STEP 5: AUDIT MISSING VALUES
# ----------------------------------------------------------------------------

missing_values = X.isna().sum()

print('\n' + '=' * 70)
print('MISSING VALUE AUDIT')
print('=' * 70)

print(missing_values)

total_missing = int(missing_values.sum())

print(f'\nTotal missing values detected: {total_missing}')


# ----------------------------------------------------------------------------
# STEP 6: IMPUTE MISSING VALUES USING FEATURE MEDIANS
# ----------------------------------------------------------------------------

if total_missing > 0:

    feature_medians = X.median()

    print('\nFeature medians used for imputation:')
    print(feature_medians)

    X_clean = X.fillna(feature_medians)

    print('\nMissing values after median imputation:')
    print(X_clean.isna().sum())

else:

    X_clean = X.copy()

    print('\nNo missing values detected. Imputation was not required.')


# ----------------------------------------------------------------------------
# STEP 7: STANDARDIZE FEATURES
# ----------------------------------------------------------------------------

scaler = StandardScaler()

X_scaled_array = scaler.fit_transform(X_clean)

X_scaled = pd.DataFrame(
    X_scaled_array,
    columns=features,
    index=X_clean.index
)


# ----------------------------------------------------------------------------
# STEP 8: VERIFY STANDARDIZATION
# ----------------------------------------------------------------------------

scaled_means = X_scaled.mean()
scaled_variances = X_scaled.var(ddof=0)

print('\n' + '=' * 70)
print('POST-SCALING FEATURE MEANS')
print('=' * 70)
print(scaled_means)

print('\n' + '=' * 70)
print('POST-SCALING FEATURE VARIANCES')
print('=' * 70)
print(scaled_variances)


# ----------------------------------------------------------------------------
# STEP 9: RAW VS. SCALED COMPARISON
# ----------------------------------------------------------------------------

comparison = pd.DataFrame({
    'raw_variance': raw_variances,
    'scaled_mean': scaled_means,
    'scaled_variance': scaled_variances
})

print('\n' + '=' * 70)
print('RAW VS. SCALED COMPARISON')
print('=' * 70)
print(comparison)


# ============================================================================
# TASK 3: UNCONSTRAINED K-MEANS TUNING
# ============================================================================

print('\n' + '=' * 70)
print('TASK 3: K-MEANS TUNING FROM K=2 TO K=10')
print('=' * 70)

k_values = list(range(2, 11))

inertia_values = []
silhouette_values = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_

    silhouette = silhouette_score(
        X_scaled,
        cluster_labels
    )

    inertia_values.append(inertia)
    silhouette_values.append(silhouette)

    print(
        f'K={k} | '
        f'Inertia={inertia:.6f} | '
        f'Silhouette Score={silhouette:.6f}'
    )


# ----------------------------------------------------------------------------
# STEP 10: CREATE METRICS TABLE
# ----------------------------------------------------------------------------

metrics_df = pd.DataFrame({
    'K': k_values,
    'Inertia': inertia_values,
    'Silhouette_Score': silhouette_values
})

print('\n' + '=' * 70)
print('K-MEANS TUNING SUMMARY')
print('=' * 70)

print(metrics_df.to_string(index=False))


# ----------------------------------------------------------------------------
# STEP 11: IDENTIFY BEST SILHOUETTE SCORE
# ----------------------------------------------------------------------------

best_index = metrics_df['Silhouette_Score'].idxmax()

best_k_silhouette = int(
    metrics_df.loc[best_index, 'K']
)

best_silhouette = metrics_df.loc[
    best_index,
    'Silhouette_Score'
]

print('\n' + '=' * 70)
print('MATHEMATICAL SILHOUETTE RESULT')
print('=' * 70)

print(f'Highest Silhouette Score: {best_silhouette:.6f}')
print(f'K with highest Silhouette Score: {best_k_silhouette}')


# ----------------------------------------------------------------------------
# STEP 12: SELECT OPERATIONAL K
# ----------------------------------------------------------------------------

selected_k = 4

print('\n' + '=' * 70)
print('SELECTED OPERATIONAL CLUSTER COUNT')
print('=' * 70)

print(f'Selected K: {selected_k}')


# ============================================================================
# TASK 4: DUAL-PANEL DIAGNOSTIC VISUALIZATION
# ============================================================================

print('\n' + '=' * 70)
print('TASK 4: GENERATING DIAGNOSTIC FIGURE')
print('=' * 70)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

selected_index = k_values.index(selected_k)

axes[0].plot(
    k_values,
    inertia_values,
    marker='o'
)

axes[0].scatter(
    selected_k,
    inertia_values[selected_index],
    s=120,
    marker='X',
    zorder=5
)

axes[0].set_title('K-Means Inertia Elbow Curve')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].set_xticks(k_values)
axes[0].grid(True, alpha=0.3)

axes[0].annotate(
    f'Selected K={selected_k}',
    xy=(
        selected_k,
        inertia_values[selected_index]
    ),
    xytext=(
        selected_k + 0.5,
        inertia_values[selected_index] + 25
    ),
    arrowprops={
        'arrowstyle': '->'
    }
)

best_silhouette_index = k_values.index(
    best_k_silhouette
)

axes[1].plot(
    k_values,
    silhouette_values,
    marker='o'
)

axes[1].scatter(
    best_k_silhouette,
    silhouette_values[best_silhouette_index],
    s=120,
    marker='X',
    zorder=5
)

axes[1].scatter(
    selected_k,
    silhouette_values[selected_index],
    s=100,
    marker='s',
    zorder=5
)

axes[1].set_title('K-Means Silhouette Score')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_xticks(k_values)
axes[1].grid(True, alpha=0.3)

axes[1].annotate(
    f'Peak K={best_k_silhouette}',
    xy=(
        best_k_silhouette,
        silhouette_values[best_silhouette_index]
    ),
    xytext=(
        best_k_silhouette - 3,
        silhouette_values[best_silhouette_index] - 0.025
    ),
    arrowprops={
        'arrowstyle': '->'
    }
)

axes[1].annotate(
    f'Selected K={selected_k}',
    xy=(
        selected_k,
        silhouette_values[selected_index]
    ),
    xytext=(
        selected_k + 0.5,
        silhouette_values[selected_index] - 0.025
    ),
    arrowprops={
        'arrowstyle': '->'
    }
)

fig.suptitle(
    'Warehouse K-Means Cluster Diagnostics - Team 3',
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f'\nFigure saved to: {FIGURE_PATH}')


# ============================================================================
# TASK 5: FINAL K-MEANS MODEL & RAW CENTROID EXTRACTION
# ============================================================================

print('\n' + '=' * 70)
print('TASK 5: FINAL MODEL & RAW CENTROID EXTRACTION')
print('=' * 70)


# ----------------------------------------------------------------------------
# STEP 13: FIT FINAL K-MEANS MODEL
# ----------------------------------------------------------------------------

final_kmeans = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=10
)

final_cluster_labels = final_kmeans.fit_predict(
    X_scaled
)

print(
    f'\nFinal K-Means model fitted successfully with K={selected_k}.'
)


# ----------------------------------------------------------------------------
# STEP 14: ADD CLUSTER LABELS BACK TO DATA
# ----------------------------------------------------------------------------

df_clustered = df.copy()

df_clustered['cluster'] = pd.Series(
    final_cluster_labels,
    index=X_scaled.index
)

print('\nCluster labels added back to original DataFrame.')

print('\nCluster counts:')
print(
    df_clustered['cluster']
    .value_counts()
    .sort_index()
)


# ----------------------------------------------------------------------------
# STEP 15: BUILD PHYSICAL DATAFRAME FOR CENTROIDS
# ----------------------------------------------------------------------------

# We use the imputed but UNSCALED physical values for centroid interpretation.
physical_features = X_clean.copy()

physical_features['cluster'] = final_cluster_labels


# ----------------------------------------------------------------------------
# STEP 16: COMPUTE RAW PHYSICAL CENTROIDS
# ----------------------------------------------------------------------------

raw_centroids = (
    physical_features
    .groupby('cluster')[features]
    .mean()
)

print('\n' + '=' * 70)
print('RAW PHYSICAL CENTROIDS')
print('=' * 70)

print(
    raw_centroids.round(3).to_string()
)


# ----------------------------------------------------------------------------
# STEP 17: VERIFY PHYSICAL UNITS
# ----------------------------------------------------------------------------

print('\nCentroid units:')

print(
    '- daily_picking_frequency: picks/day'
)

print(
    '- unit_weight_kg: kilograms'
)

print(
    '- humidity_tolerance_pct: percent'
)

print(
    '- shelf_life_days: days'
)


# ----------------------------------------------------------------------------
# STEP 18: FINAL STATUS
# ----------------------------------------------------------------------------

print('\n' + '=' * 70)
print('TASK 5 COMPLETE')
print('=' * 70)

print(
    '\nThe final cluster labels are index-aligned with the original '
    'dataset, and the centroid table is expressed in original physical '
    'units rather than standardized coordinates.'
)