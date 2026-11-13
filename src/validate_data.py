import os
import pandas as pd
import numpy as np  
# Resolve the directory where this validate_data.py script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Point dynamically to the corrupted dataset inside the raw data folder
RAW_DATA_PATH = os.path.join(
    SCRIPT_DIR,
    '..',
    'data',
    'raw',
    'solar_telemetry_corrupted.csv'
)

# Directory where the processed files will be saved
PROCESSED_DIR = os.path.join(
    SCRIPT_DIR,
    '..',
    'data',
    'processed'
)

# Create the processed directory if it does not exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load the corrupted CSV file into a pandas DataFrame
if os.path.exists(RAW_DATA_PATH):
    df = pd.read_csv(RAW_DATA_PATH)

    print('Ingestion Successful!')
    print('Loaded:', df.shape[0], 'rows and', df.shape[1], 'columns.\n')

    print('Initial data preview:')
    print(df.head(3))
    print('-' * 60)

else:
    print('Error: Dataset file not found at:', RAW_DATA_PATH)
    exit()
    # ============================================================
# AUDIT A: PRIMARY KEY DUPLICATES
# ============================================================

# Check for duplicated timestamp + panel_id combinations
duplicate_mask = df.duplicated(
    subset=['timestamp', 'panel_id'],
    keep=False
)

df_duplicates = df[duplicate_mask]

print('--- AUDIT A: PRIMARY KEY DUPLICATE SCAN ---')
print(f'Detected {len(df_duplicates)} duplicate rows.')

if len(df_duplicates) > 0:
    print('Duplicate rows identified (displaying first 4):')
    print(
        df_duplicates[
            ['timestamp', 'panel_id', 'voltage_v', 'current_a']
        ].head(4)
    )

print('-' * 60)
# ============================================================
# AUDIT B: NEGATIVE ELECTRICAL VALUES
# ============================================================

# Identify negative voltage and current values
voltage_violations = df[df['voltage_v'] < 0]
current_violations = df[df['current_a'] < 0]

print('--- AUDIT B: ELECTRICAL VIOLATION SCAN ---')
print(f'Detected {len(voltage_violations)} rows with negative voltage.')
print(f'Detected {len(current_violations)} rows with negative current.')

if len(voltage_violations) > 0:
    print('Example negative voltage records:')
    print(
        voltage_violations[
            ['timestamp', 'panel_id', 'voltage_v']
        ].head(3)
    )

if len(current_violations) > 0:
    print('Example negative current records:')
    print(
        current_violations[
            ['timestamp', 'panel_id', 'current_a']
        ].head(3)
    )

print('-' * 60)
# ============================================================
# AUDIT C: EFFICIENCY BOUNDS CHECK
# ============================================================

# Identify efficiency values outside the valid 0-100% range
efficiency_violations = df[
    (df['efficiency_pct'] < 0) |
    (df['efficiency_pct'] > 100)
]

print('--- AUDIT C: MECHANICAL RANGE VIOLATION SCAN ---')
print(
    f'Detected {len(efficiency_violations)} rows '
    'with out-of-bounds efficiency.'
)

if len(efficiency_violations) > 0:
    print('Example efficiency violations:')
    print(
        efficiency_violations[
            ['timestamp', 'panel_id', 'efficiency_pct']
        ].head(3)
    )

print('-' * 60)
# ============================================================
# AUDIT D: TEMPORAL LOGIC & TEMPERATURE SPIKES
# ============================================================

# Sort records by panel and timestamp
df_sorted = df.sort_values(
    by=['panel_id', 'timestamp']
).copy()

# Calculate the absolute temperature change between consecutive records
# for each panel independently
df_sorted['temp_change'] = (
    df_sorted.groupby('panel_id')['temperature_c']
    .diff()
    .abs()
)

# Identify temperature changes greater than 30°C
temp_violations = df_sorted[
    df_sorted['temp_change'] > 30.0
]

print('--- AUDIT D: TEMPORAL SENSOR SPIKE SCAN ---')
print(
    f'Detected {len(temp_violations)} rapid hourly '
    'temperature fluctuations (>30°C/hr).'
)

if len(temp_violations) > 0:
    print('Example malfunctioning sensor records:')
    print(
        temp_violations[
            ['timestamp', 'panel_id', 'temperature_c', 'temp_change']
        ].head(3)
    )

print('-' * 60)
# ============================================================
# AUTOMATED DATA QUALITY GATE
# ============================================================

# Collect all corrupted row indices from the four audits
corrupted_indices = set()

corrupted_indices.update(
    duplicate_mask[duplicate_mask].index
)

corrupted_indices.update(
    voltage_violations.index
)

corrupted_indices.update(
    current_violations.index
)

corrupted_indices.update(
    efficiency_violations.index
)

corrupted_indices.update(
    temp_violations.index
)

# Separate clean and corrupted records
df_corrupted = df.loc[list(corrupted_indices)].copy()
df_clean = df.drop(index=list(corrupted_indices)).copy()

# Print final summary
print('=== AUTOMATED DATA QUALITY GATE SUMMARY ===')
print(f'Total Rows Audited: {len(df)}')

print(
    f'Approved Clean Rows: {len(df_clean)} '
    f'({len(df_clean) / len(df) * 100:.2f}%)'
)

print(
    f'Isolated Corrupted Rows: {len(df_corrupted)} '
    f'({len(df_corrupted) / len(df) * 100:.2f}%)'
)

print('============================================\n')

# Define output paths
CLEAN_OUT_PATH = os.path.join(
    PROCESSED_DIR,
    'solar_telemetry_clean.csv'
)

ANOMALY_OUT_PATH = os.path.join(
    PROCESSED_DIR,
    'solar_telemetry_anomalies.csv'
)

# Export both datasets
df_clean.to_csv(
    CLEAN_OUT_PATH,
    index=False
)

df_corrupted.to_csv(
    ANOMALY_OUT_PATH,
    index=False
)

print(f'Successfully exported clean data to: {CLEAN_OUT_PATH}')
print(f'Successfully exported anomalies subset to: {ANOMALY_OUT_PATH}')