import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title='Segmentacion de Envios - E-Commerce', page_icon='📦', layout='centered')

@st.cache_resource
def load_artifacts():
    knn = joblib.load('knn_cluster_model.joblib')
    scaler = joblib.load('scaler_cluster.joblib')
    profiles = joblib.load('cluster_profiles.joblib')
    variables = joblib.load('cluster_vars.joblib')
    return knn, scaler, profiles, variables

knn_model, scaler_cluster, cluster_profiles, cluster_vars = load_artifacts()

st.title('📦 Clasificador de Segmento Logístico')
st.write(
    'Ingresa los datos de un nuevo envío para obtener, en tiempo real, su segmento '
    'operativo (obtenido originalmente con K-Means) y la recomendación logística asociada.'
)

with st.form('shipment_form'):
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input('Peso del paquete (gramos)', min_value=500, max_value=10000, value=3000, step=50)
        cost = st.number_input('Costo del producto ($)', min_value=50, max_value=400, value=200, step=5)
        discount = st.number_input('Descuento ofrecido (%)', min_value=0, max_value=70, value=10, step=1)
    with col2:
        calls = st.number_input('Llamadas al servicio al cliente', min_value=0, max_value=10, value=4, step=1)
        purchases = st.number_input('Compras previas del cliente', min_value=1, max_value=12, value=3, step=1)
        rating = st.number_input('Calificación del cliente (1-5)', min_value=1, max_value=5, value=3, step=1)
    submitted = st.form_submit_button('Clasificar envío')

if submitted:
    X_new = np.array([[weight, cost, discount, calls, purchases, rating]])
    X_new_scaled = scaler_cluster.transform(X_new)
    cluster_pred = int(knn_model.predict(X_new_scaled)[0])
    profile = cluster_profiles[cluster_pred]

    st.success(f'Segmento asignado: Clúster {cluster_pred} — {profile["label"]}')
    st.metric('Tasa histórica de retraso en este segmento', f'{profile["tasa_retraso_historica"] * 100:.1f}%')
    st.info(f'Recomendación operativa: {profile["recomendacion"]}')

    with st.expander('Ver perfil promedio del segmento'):
        st.json(profile['perfil_promedio'])

st.caption('Modelo K-NN entrenado sobre segmentos generados con K-Means. Taller 6 - ICYA3004, Universidad de los Andes.')
