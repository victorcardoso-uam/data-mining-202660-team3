import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def run_feature_engineering_pipeline():

    # =========================================================================
    # PATH CONFIGURATION
    # =========================================================================

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    RAW_DIR = os.path.join(
        SCRIPT_DIR,
        "..",
        "data",
        "raw"
    )

    PROCESSED_DIR = os.path.join(
        SCRIPT_DIR,
        "..",
        "data",
        "processed"
    )


    TELEMETRY_PATH = os.path.join(
        RAW_DIR,
        "solar_telemetry_cleaned.csv"
    )

    METADATA_PATH = os.path.join(
        RAW_DIR,
        "panel_metadata.csv"
    )

    WEATHER_PATH = os.path.join(
        RAW_DIR,
        "weather_station_log.csv"
    )


    # =========================================================================
    # STEP 1: LOADING & JOINING DATA
    # =========================================================================

    print("\n--- STEP 1: LOADING & JOINING DATA ---")


    df_telemetry = pd.read_csv(
        TELEMETRY_PATH
    )

    df_metadata = pd.read_csv(
        METADATA_PATH
    )

    df_weather = pd.read_csv(
        WEATHER_PATH
    )


    print(
        "Telemetry shape before duplicate removal:",
        df_telemetry.shape
    )

    print(
        "Duplicate telemetry rows:",
        df_telemetry.duplicated().sum()
    )


    # -------------------------------------------------------------------------
    # Compatibility step between Activity 3 and Activity 4
    # Activity 3 preserves 722 rows.
    # Activity 4 expects 720 telemetry observations.
    # -------------------------------------------------------------------------

    df_telemetry = (
        df_telemetry
        .drop_duplicates()
        .reset_index(drop=True)
    )


    print(
        "Raw Telemetry shape:",
        df_telemetry.shape
    )

    print(
        "Raw Metadata shape:",
        df_metadata.shape
    )

    print(
        "Raw Weather shape:",
        df_weather.shape
    )


    # Convert timestamps to datetime
    df_telemetry["timestamp"] = pd.to_datetime(
        df_telemetry["timestamp"]
    )

    df_weather["timestamp"] = pd.to_datetime(
        df_weather["timestamp"]
    )


    # Merge telemetry with metadata
    df_merged = pd.merge(
        df_telemetry,
        df_metadata,
        on="panel_id",
        how="left"
    )


    # Merge previous result with weather
    df_final = pd.merge(
        df_merged,
        df_weather,
        on="timestamp",
        how="left"
    )


    print(
        "Post-Merge shape:",
        df_final.shape
    )


    assert df_final.shape == (720, 14), (
        "Error: Expected final merge shape "
        "(720, 14)"
    )


    # =========================================================================
    # STEP 2: ONE-HOT CATEGORICAL ENCODING
    # =========================================================================

    print("\n--- STEP 2: CATEGORICAL ENCODING ---")


    df_encoded = pd.get_dummies(
        df_final,
        columns=["manufacturer"],
        drop_first=True,
        dtype=int
    )


    print(
        "Encoded DataFrame columns:"
    )

    print(
        df_encoded.columns.tolist()
    )


    print(
        "Encoded shape:",
        df_encoded.shape
    )


    assert df_encoded.shape == (720, 15), (
        "Error: Expected encoded shape "
        "(720, 15)"
    )


    # =========================================================================
    # STEP 3: FEATURE SCALING
    # =========================================================================

    print("\n--- STEP 3: FEATURE SCALING ---")


    # -------------------------------------------------------------------------
    # StandardScaler
    # -------------------------------------------------------------------------

    cols_to_standardize = [
        "voltage_v",
        "ambient_temp_c"
    ]


    scaler_std = StandardScaler()


    scaled_std = scaler_std.fit_transform(
        df_encoded[
            cols_to_standardize
        ]
    )


    df_encoded[
        "voltage_v_scaled"
    ] = scaled_std[:, 0]


    df_encoded[
        "ambient_temp_c_scaled"
    ] = scaled_std[:, 1]


    print(
        "StandardScaler Means:"
    )

    print(
        scaler_std.mean_
    )


    print(
        "StandardScaler Standard Deviations:"
    )

    print(
        np.sqrt(
            scaler_std.var_
        )
    )


    # -------------------------------------------------------------------------
    # MinMaxScaler
    # -------------------------------------------------------------------------

    scaler_minmax = MinMaxScaler()


    scaled_minmax = scaler_minmax.fit_transform(
        df_encoded[
            ["efficiency_pct"]
        ]
    )


    df_encoded[
        "efficiency_pct_scaled"
    ] = scaled_minmax[:, 0]


    print(
        "MinMax Data Min:"
    )

    print(
        scaler_minmax.data_min_
    )


    print(
        "MinMax Data Max:"
    )

    print(
        scaler_minmax.data_max_
    )


    # =========================================================================
    # COORDINATES VERIFICATION
    # =========================================================================

    print(
        "\n--- COORDINATES VERIFICATION CHECKS ---"
    )


    print("\nRow 0:")

    print(
        "voltage_v_scaled:",
        f"{df_encoded.loc[0, 'voltage_v_scaled']:.6f}"
    )

    print(
        "ambient_temp_c_scaled:",
        f"{df_encoded.loc[0, 'ambient_temp_c_scaled']:.6f}"
    )

    print(
        "efficiency_pct_scaled:",
        f"{df_encoded.loc[0, 'efficiency_pct_scaled']:.6f}"
    )


    print("\nRow 500:")

    print(
        "voltage_v_scaled:",
        f"{df_encoded.loc[500, 'voltage_v_scaled']:.6f}"
    )

    print(
        "ambient_temp_c_scaled:",
        f"{df_encoded.loc[500, 'ambient_temp_c_scaled']:.6f}"
    )

    print(
        "efficiency_pct_scaled:",
        f"{df_encoded.loc[500, 'efficiency_pct_scaled']:.6f}"
    )


    # =========================================================================
    # STEP 4: EXPORT PROCESSED FEATURES
    # =========================================================================

    print(
        "\n--- STEP 4: EXPORTING CLEAN DATASET ---"
    )


    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )


    OUTPUT_FILE_PATH = os.path.join(
        PROCESSED_DIR,
        "solar_features_engineered.csv"
    )


    df_encoded.to_csv(
        OUTPUT_FILE_PATH,
        index=False
    )


    print(
        "Features successfully engineered and written to:",
        OUTPUT_FILE_PATH
    )


    print(
        "Final engineered dataset shape:",
        df_encoded.shape
    )


    print(
        "\nPipeline compilation completed successfully!"
    )


if __name__ == "__main__":
    run_feature_engineering_pipeline()