"""Numerically check that the app presets trigger the expected alerts with the real model."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "exam2_team3_ev_battery.csv")
meta = json.loads((ROOT / "models" / "metadata.json").read_text())
model = joblib.load(ROOT / "models" / "best_model.joblib")
scaler = joblib.load(ROOT / "models" / "scaler.joblib")
y = df["battery_cell_temp_c"]
cool, crit = round(y.quantile(.75)), round(y.quantile(.90))


def q(c, p):
    return round(float(df[c].quantile(p)), 2)


presets = {
    "Normal scenario": ({"charge_rate_c": q("charge_rate_c", .5), "discharge_rate_c": q("discharge_rate_c", .1),
                         "ambient_temp_c": q("ambient_temp_c", .1), "coolant_flow_l_min": q("coolant_flow_l_min", .9),
                         "internal_resistance_mOhm": q("internal_resistance_mOhm", .5)}, "NORMAL"),
    "High-risk scenario": ({"charge_rate_c": q("charge_rate_c", .5), "discharge_rate_c": q("discharge_rate_c", .95),
                            "ambient_temp_c": q("ambient_temp_c", .95), "coolant_flow_l_min": q("coolant_flow_l_min", .05),
                            "internal_resistance_mOhm": q("internal_resistance_mOhm", .95)}, "CRITICAL"),
}
print(f"Thresholds: active cooling >= {cool} C (P75={y.quantile(.75):.2f}), "
      f"critical >= {crit} C (P90={y.quantile(.9):.2f})")
rows, ok = [], True
for name, (vals, expected) in presets.items():
    X = pd.DataFrame([vals])[meta["features"]]
    if meta["requires_scaler"]:
        X = scaler.transform(X)
    p = float(model.predict(X)[0])
    p = float(np.exp(p)) if meta["log_target"] else p
    level = "CRITICAL" if p >= crit else "ACTIVE COOLING" if p >= cool else "NORMAL"
    ok &= level == expected
    print(f"{name}: {vals} -> {p:.2f} C -> {level} (expected {expected}) {'OK' if level == expected else 'FAIL'}")
    rows.append({"Scenario": name, **vals, "Predicted_C": round(p, 2), "Alert": level})
pd.DataFrame(rows).to_csv(ROOT / "reports" / "task3_presets.csv", index=False)
sys.exit(0 if ok else 1)
