from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Ubicación del CSV respecto al script.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "pruning_team3_ev_battery.csv"

df = pd.read_csv(DATA_PATH)

# Separamos las variables predictoras de la variable objetivo.
TARGET = "thermal_runaway_risk"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Creamos una única división para todo el experimento.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("ACTIVIDAD 13 - EQUIPO 3 - VICTORIA MORALES")
print(f"\nTotal de registros: {len(df)}")
print(f"Variables predictoras: {X.columns.tolist()}")
print(f"Registros de entrenamiento: {len(X_train)}")
print(f"Registros de prueba: {len(X_test)}")

print("\nClases en entrenamiento:")
print(y_train.value_counts().sort_index())

print("\nClases en prueba:")
print(y_test.value_counts().sort_index())

from sklearn.tree import DecisionTreeClassifier

# Candidatos de la tabla de la actividad.
ALPHAS = [0.001, 0.015, 0.080]
resultados = []

for alpha in ALPHAS:
    clf = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=alpha,
    )
    clf.fit(X_train, y_train)

    hojas = clf.get_n_leaves()
    profundidad = clf.get_depth()
    error_train = 1.0 - clf.score(X_train, y_train)
    error_test = 1.0 - clf.score(X_test, y_test)

    # Costo solicitado por el laboratorio: errores en escala 0–1.
    costo_total = error_test + alpha * hojas

    resultados.append({
        "alpha": alpha,
        "hojas": hojas,
        "profundidad": profundidad,
        "error_train": error_train,
        "error_test": error_test,
        "costo_total": costo_total,
    })

tabla = pd.DataFrame(resultados)

# Los porcentajes se usan solo para mostrar los errores.
tabla_visible = tabla.copy()
tabla_visible["error_train"] = tabla["error_train"].map(
    lambda valor: f"{valor:.2%}"
)
tabla_visible["error_test"] = tabla["error_test"].map(
    lambda valor: f"{valor:.2%}"
)

print("\nRESULTADOS DEL EXPERIMENTO")
print(tabla_visible.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

print("\nALPHA(S) CON MENOR ERROR DE PRUEBA:")
print(
    tabla.loc[
        tabla["error_test"] == tabla["error_test"].min(), "alpha"
    ].tolist()
)

print("\nALPHA(S) CON MENOR COSTO TOTAL:")
print(
    tabla.loc[
        tabla["costo_total"] == tabla["costo_total"].min(), "alpha"
    ].tolist()
)

# Comprobación adicional por la discrepancia 0.000 / 0.001.
clf_sin_poda = DecisionTreeClassifier(random_state=42, ccp_alpha=0.0)
clf_sin_poda.fit(X_train, y_train)

print("\nCOMPROBACIÓN ADICIONAL: ALPHA = 0.000")
print(f"Hojas: {clf_sin_poda.get_n_leaves()}")
print(f"Profundidad: {clf_sin_poda.get_depth()}")
print(f"Error de entrenamiento: {1 - clf_sin_poda.score(X_train, y_train):.2%}")
print(f"Error de prueba: {1 - clf_sin_poda.score(X_test, y_test):.2%}")

# Guardamos los resultados numéricos sin redondear.
REPORTS_DIR = BASE_DIR / "reports" / "activity_13_victoria_morales"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = REPORTS_DIR / "pruning_results.csv"
tabla.to_csv(RESULTS_PATH, index=False)

print(f"\nTabla guardada en: {RESULTS_PATH.relative_to(BASE_DIR)}")