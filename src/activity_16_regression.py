from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "regression_team3_ev_battery.csv"

df = pd.read_csv(DATA_PATH)

print("Dimensiones:", df.shape)
print("\nColumnas:")
print(df.columns.tolist())
print("\nPrimeras cinco filas:")
print(df.head())
print("\nValores faltantes:")
print(df.isna().sum())





from sklearn.model_selection import train_test_split

TARGET = "battery_cell_temp_c"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nRegistros de entrenamiento:", len(X_train))
print("Registros de prueba:", len(X_test))




feature_order = [
    "discharge_current_a",
    "ambient_temp_c",
    "internal_resistance_mohm",
    "coolant_flow_l_min",
    "state_of_charge_pct",
]

model_features = {
    f"Model {i}": feature_order[:i]
    for i in range(1, 6)
}

for name, features in model_features.items():
    print(f"\n{name}: {features}")





from sklearn.linear_model import LinearRegression

trained_models = {}
train_predictions = {}
test_predictions = {}

for name, features in model_features.items():
    model = LinearRegression()

    model.fit(X_train[features], y_train)

    trained_models[name] = model
    train_predictions[name] = model.predict(X_train[features])
    test_predictions[name] = model.predict(X_test[features])

    print(
        f"{name}: entrenado con {len(features)} variables; "
        f"{len(test_predictions[name])} predicciones de prueba."
    )





from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

results = []

for name, features in model_features.items():
    train_pred = train_predictions[name]
    test_pred = test_predictions[name]

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    n = len(y_train)
    p = len(features)
    adjusted_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)

    results.append({
        "Model": name,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Adjusted R2": adjusted_r2,
        "MAE": mean_absolute_error(y_test, test_pred),
        "RMSE": mean_squared_error(y_test, test_pred) ** 0.5,
    })

comparison = pd.DataFrame(results)

print("\nCOMPARACIÓN DE MODELOS")
print(comparison.to_string(index=False, float_format="{:.4f}".format))





FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

top_models = comparison.nsmallest(2, "RMSE")["Model"]

for name in top_models:
    predicted = test_predictions[name]
    residuals = y_test.to_numpy() - predicted

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(predicted, residuals, alpha=0.65)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Residuals vs. Predicted Values")
    axes[0].set_xlabel("Predicted Cell Temperature (°C)")
    axes[0].set_ylabel("Residual: Actual - Predicted (°C)")
    axes[0].grid(alpha=0.25)

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q Plot")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"EV Battery — {name} — Test Residuals")
    fig.tight_layout()

    filename = f"activity_16_{name.replace(' ', '_').lower()}.png"
    fig.savefig(FIG_DIR / filename, dpi=180)
    plt.close(fig)

    print(f"Gráfico guardado: {FIG_DIR / filename}")