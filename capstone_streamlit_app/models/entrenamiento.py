import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# 1. Crear carpeta si no existe
if not os.path.exists('models'):
    os.makedirs('models')

# 2. Cargar el dataset crudo
df_raw = pd.read_csv("data/UrbanMart_Transactions.csv")

# ==========================================
# 3. PREPROCESAMIENTO: Generar variables RFM y Churn
# ==========================================
# Convertir fechas para cálculos
df_raw['TransactionDate'] = pd.to_datetime(df_raw['TransactionDate'])
max_date = df_raw['TransactionDate'].max()

# Agrupar datos por Cliente (RFM)
df = df_raw.groupby('CustomerID').agg(
    recency=('TransactionDate', lambda x: (max_date - x.max()).days),
    frequency=('TransactionID', 'count'),
    monetary=('TransactionValue', 'sum'),
    CustomerAgeGroup=('CustomerAgeGroup', 'first')
).reset_index()

# Convertir rangos de edad a valores numéricos
age_map = {'18-25': 22, '26-35': 30, '36-50': 43, '>50': 55}
df['age'] = df['CustomerAgeGroup'].map(age_map)

# Crear Target: churn_risk_flag (1 si el cliente lleva > 40 días sin comprar)
df['churn_risk_flag'] = (df['recency'] > 40).astype(int)

# ==========================================
# 4. FASE DE ENTRENAMIENTO Y SERIALIZACIÓN
# ==========================================
X = df[['recency', 'monetary', 'frequency', 'age']]
y = df['churn_risk_flag']

# Escalar y guardar scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.joblib')
print("✅ scaler.joblib creado")

# Modelo 1: Logística
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_scaled, y)
joblib.dump(log_reg, 'models/logistic_regression.joblib')
print("✅ logistic_regression.joblib creado")

# Modelo 2: Red Neuronal
ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
ann.fit(X_scaled, y)
joblib.dump(ann, 'models/ann_model.joblib')
print("✅ ann_model.joblib creado")

# Modelo 3: RFM K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
joblib.dump(kmeans, 'models/kmeans_rfm.joblib')
print("✅ kmeans_rfm.joblib creado")

print("🎉 ¡Preprocesamiento exitoso y modelos serializados!")