"""Shared paths, data loading and metrics for Exam 2 (Team 3)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "exam2_team3_ev_battery.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
SHOTS = REPORTS / "screenshots"

FEATURES = ["charge_rate_c", "discharge_rate_c", "ambient_temp_c",
            "coolant_flow_l_min", "internal_resistance_mOhm"]
TARGET = "battery_cell_temp_c"


def load_data():
    df = pd.read_csv(DATA)
    return df


def split(df):
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def metrics(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return {
        "R2": r2,
        "Adj_R2": 1 - (1 - r2) * (n - 1) / (n - p - 1),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "p": p,
    }
