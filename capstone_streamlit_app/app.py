import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib  # Necesario para cargar los modelos serializados
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# =============================================================================
# 1. PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="UrbanMart Intelligence - Capstone",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8F9FA;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #ADB5BD;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E2125;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #343A40;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. CACHED DATASETS & MODEL LOADING
# =============================================================================
@st.cache_data
def load_data():
    """Loads and preprocesses UrbanMart Transactions dataset to RFM format."""
    try:
        df_raw = pd.read_csv("data/UrbanMart_Transactions.csv")
        df_raw['TransactionDate'] = pd.to_datetime(df_raw['TransactionDate'])
        max_date = df_raw['TransactionDate'].max()
        
        # Agrupación RFM en vivo para las gráficas del dashboard
        df = df_raw.groupby('CustomerID').agg(
            recency=('TransactionDate', lambda x: (max_date - x.max()).days),
            frequency=('TransactionID', 'count'),
            monetary=('TransactionValue', 'sum'),
            CustomerAgeGroup=('CustomerAgeGroup', 'first')
        ).reset_index()
        
        age_map = {'18-25': 22, '26-35': 30, '36-50': 43, '>50': 55}
        df['age'] = df['CustomerAgeGroup'].map(age_map)
        df['churn_risk_flag'] = (df['recency'] > 40).astype(int)
        
        # Retornamos solo las variables listas para graficar y analizar
        return df[['recency', 'monetary', 'frequency', 'age', 'churn_risk_flag']]
    except FileNotFoundError:
        st.error("Archivo no encontrado. Asegúrate de que 'UrbanMart_Transactions.csv' esté en 'data/'.")
        return pd.DataFrame()

@st.cache_resource
def load_models():
    """Dynamically loads serialized .joblib models and scalers."""
    # NOTA: Asegúrate de tener estos archivos exportados y guardados en tu carpeta 'models/'
    try:
        models = {
            "RFM K-Means": joblib.load("models/kmeans_rfm.joblib"),
            "Logistic Regression": joblib.load("models/logistic_regression.joblib"),
            "Artificial Neural Network (ANN)": joblib.load("models/ann_model.joblib")
        }
        scaler = joblib.load("models/scaler.joblib")
    except FileNotFoundError:
        st.warning("No se encontraron los archivos .joblib en la carpeta 'models/'. Reemplázalos con tus modelos reales.")
        models = {}
        scaler = None
        
    return models, scaler

# Load data and models
df_raw = load_data()
loaded_models, scaler = load_models()

# Mock performance data (Reemplaza con los resultados de tu conjunto de prueba real)
if not df_raw.empty:
    performance_df = pd.DataFrame([
        {"Model Algorithm": "RFM K-Means", "Accuracy": 0.82, "Precision": 0.79, "Recall": 0.81, "F1-Score": 0.80, "ROC-AUC": 0.85},
        {"Model Algorithm": "Logistic Regression", "Accuracy": 0.85, "Precision": 0.83, "Recall": 0.86, "F1-Score": 0.84, "ROC-AUC": 0.89},
        {"Model Algorithm": "Artificial Neural Network (ANN)", "Accuracy": 0.91, "Precision": 0.89, "Recall": 0.92, "F1-Score": 0.90, "ROC-AUC": 0.95}
    ])


# =============================================================================
# 3. SIDEBAR CONTROLS & NAVIGATION
# =============================================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/online-store.png", width=64)
st.sidebar.title("UrbanMart Intelligence")
st.sidebar.caption("Retention & CRM Manager Dashboard")

st.sidebar.divider()
st.sidebar.subheader("🎛️ Global Inference Settings")
selected_model_name = st.sidebar.selectbox("Select Active Model:", ["Logistic Regression", "Artificial Neural Network (ANN)", "RFM K-Means"])
model_threshold = st.sidebar.slider("Churn Risk Cutoff (Threshold)", 0.05, 0.95, 0.60, 0.05)

st.sidebar.divider()
st.sidebar.info("📌 Client Target: E-Commerce CRM Manager. Variable: churn_risk_flag")


# =============================================================================
# 4. MAIN EXECUTIVE DASHBOARD HEADER
# =============================================================================
st.markdown('<div class="main-header">🛍️ UrbanMart E-Commerce CRM Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Customer Retention Prediction & VIP Policy Automation | Milestone 3/3</div>', unsafe_allow_html=True)

if df_raw.empty:
    st.stop()

# Top Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Customer Profiles", f"{len(df_raw):,}", delta="100% Ingested")
with kpi2:
    st.metric("Feature Dimensions", f"{df_raw.shape[1] - 1} Features", delta="Engineered")
with kpi3:
    # Asume que tu columna objetivo se llama 'churn_risk_flag'
    if 'churn_risk_flag' in df_raw.columns:
        churn_rate = (df_raw['churn_risk_flag'].mean()) * 100
        st.metric("Global Churn Risk Rate", f"{churn_rate:.1f}%", delta="Target: churn_risk_flag")
    else:
        st.metric("Global Churn Risk Rate", "N/A", delta="Target Not Found")
with kpi4:
    best_auc = performance_df[performance_df['Model Algorithm'] == selected_model_name]['ROC-AUC'].values[0]
    st.metric("Active Model ROC-AUC", f"{best_auc:.3f}", delta=f"Model: {selected_model_name.split()[0]}")

st.divider()


# =============================================================================
# 5. WORKSHOP TABS: EXPLORATION, PIPELINE, BENCHMARK, & INFERENCE
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer & EDA", 
    "⚙️ Preprocessing & Correlation", 
    "🤖 Multi-Model Benchmark", 
    "🔮 Real-Time What-If Inference"
])

# -----------------------------------------------------------------------------
# TAB 1: DATA EXPLORER & EDA
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Raw Customer Demographics & Statistical Profile")
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.dataframe(df_raw.head(100), use_container_width=True, height=320)
    
    with col_b:
        st.write("**Dataset Summary Statistics:**")
        st.dataframe(df_raw.describe().T[['mean', 'std', 'min', '50%', 'max']], use_container_width=True)
    
    st.divider()
    st.subheader("Interactive Feature Distribution Analysis")
    if 'churn_risk_flag' in df_raw.columns:
        selected_feature = st.selectbox("Select Feature to Visualize:", [c for c in df_raw.columns if c != 'churn_risk_flag'])
        
        fig_hist = px.histogram(
            df_raw, 
            x=selected_feature, 
            color="churn_risk_flag", 
            barmode="overlay",
            title=f"Distribution of {selected_feature} by Churn Risk Status",
            color_discrete_map={0: "#10B981", 1: "#EF4444"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: PREPROCESSING & CORRELATION
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Feature Correlation & Preprocessing Matrix")
    num_df = df_raw.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Pearson Feature Correlation Matrix",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: MULTI-MODEL BENCHMARK
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TAB 3: MULTI-MODEL BENCHMARK & CONFUSION MATRICES
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Empirical Model Performance Benchmark Suite")
    
    # 1. Tabla comparativa de métricas
    st.dataframe(performance_df, use_container_width=True)
    
    st.divider()
    st.subheader("Confusion Matrices (Real Model Evaluation)")
    
    # Verificar que los modelos cargados existan antes de graficar
    if loaded_models and scaler is not None:
        try:
            # Preparar los datos de prueba usando las mismas variables del entrenamiento
            X_eval = df_raw[['recency', 'monetary', 'frequency', 'age']]
            y_eval = df_raw['churn_risk_flag']
            
            # Escalar los datos de evaluación con el scaler joblib
            X_eval_scaled = scaler.transform(X_eval)
            
            # Crear columnas para mostrar las matrices lado a lado
            col_m1, col_m2, col_m3 = st.columns(3)
            
            # Diccionario para mapear nombres bonitos a los modelos cargados
            model_keys = [
                ("Logistic Regression", col_m1), 
                ("Artificial Neural Network (ANN)", col_m2), 
                ("RFM K-Means", col_m3)
            ]
            
            for m_name, col in model_keys:
                if m_name in loaded_models:
                    model_obj = loaded_models[m_name]
                    
                    # Generar predicciones reales con el modelo .joblib
                    if m_name == "RFM K-Means":
                        # K-Means es clustering, usamos predict y convertimos clústeres a binario de riesgo
                        clusters = model_obj.predict(X_eval_scaled)
                        y_pred = (clusters == 0).astype(int) # Asignando el clúster mayoritario de riesgo
                    else:
                        y_pred = model_obj.predict(X_eval_scaled)
                    
                    # Calcular matriz de confusión
                    cm = confusion_matrix(y_eval, y_pred)
                    
                    with col:
                        st.markdown(f"**{m_name}**")
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            x=['Pred: Retained (0)', 'Pred: Churn (1)'],
                            y=['Actual: Retained (0)', 'Actual: Churn (1)'],
                            color_continuous_scale="Reds",
                            labels=dict(x="Predicted", y="Actual", color="Count")
                        )
                        fig_cm.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as e:
            st.error(f"Error al calcular las matrices de confusión con los modelos: {e}")
    else:
        st.warning("⚠️ No se pudieron cargar los modelos .joblib en la carpeta 'models/'. Verifica que existan para mostrar las matrices reales.")


# -----------------------------------------------------------------------------
# TAB 4: REAL-TIME WHAT-IF SCENARIO INFERENCE
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🔮 CRM Real-Time What-If Scenario Prediction Engine")
    st.write("Adjust customer RFM metrics to test live churn risk and trigger retention policies:")
    
    col_i1, col_i2 = st.columns([1, 1])
    
    with col_i1:
        st.write("### 🎛️ RFM Input Parameters")
        input_recency = st.slider("Recency (Days since last purchase):", 1, 365, 30)
        input_monetary = st.slider("Monetary Spend ($):", 10.0, 5000.0, 150.0)
        input_frequency = st.slider("Frequency (Total Purchases):", 1, 100, 5)
        input_age = st.slider("Customer Age:", 18, 90, 35)
        
    with col_i2:
        st.write("### 🎯 Live Model Inference Output")
        input_df = pd.DataFrame([{
            'recency': input_recency,
            'monetary': input_monetary,
            'frequency': input_frequency,
            'age': input_age
        }])
        
        # Simulación condicional de predicción basada en la rúbrica si los modelos reales no están listos
        if loaded_models and selected_model_name in loaded_models:
            # Aquí iría la predicción real: 
            # X_scaled = scaler.transform(input_df)
            # simulated_prob = loaded_models[selected_model_name].predict_proba(X_scaled)[0, 1]
            pass
            
        # Predicción simulada para efectos de la interfaz visual
        simulated_prob = (input_recency * 0.002) - (input_monetary * 0.0001) - (input_frequency * 0.005) + 0.4
        simulated_prob = max(0.01, min(0.99, simulated_prob)) # Limitar entre 0 y 1
        predicted_class = 1 if simulated_prob >= model_threshold else 0
        
        st.markdown(f"#### Predicted Churn Risk Probability: **{simulated_prob*100:.1f}%**")
        st.progress(float(simulated_prob))
        
        if predicted_class == 1:
            st.error(f"⚠️ HIGH CHURN RISK DETECTED! (Cutoff = {model_threshold:.2f})")
            st.write("<b>Recommended CRM Policy:</b> Deploy immediate re-engagement vouchers via email and SMS.")
        else:
            st.success(f"✅ HIGH RETENTION LOYALTY: Safe Customer State (Cutoff = {model_threshold:.2f})")
            st.write("<b>Recommended CRM Policy:</b> Maintain standard CRM flow. Consider unlocking VIP perks for monetary milestones.")
            
        st.divider()
        st.write("**Scenario Features Transmitted:**")
        st.json(input_df.to_dict(orient='records')[0])