import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title='Caso de Estudio - Breast Cancer', page_icon='🔬', layout='wide')

@st.cache_resource
def load_artifacts():
    scaler = joblib.load('bc_scaler.joblib')
    pca = joblib.load('bc_pca.joblib')
    knn = joblib.load('bc_knn_model.joblib')
    profiles = joblib.load('bc_cluster_profiles.joblib')
    feature_names = joblib.load('bc_feature_names.joblib')
    return scaler, pca, knn, profiles, feature_names

scaler_bc, pca_bc, knn_bc, cluster_profiles, feature_names = load_artifacts()

st.title('🔬 Caso de Estudio: Segmentación de Biopsias (Breast Cancer Wisconsin)')
st.caption(
    'Demostración académica del flujo K-Means + K-NN (Fase IV, Taller 6, ICYA3004). '
    'NO constituye una herramienta de diagnóstico clínico real.'
)
st.warning(
    'Uso exclusivamente académico: este modelo se entrenó sobre un dataset canónico de '
    'referencia (scikit-learn) y no reemplaza ningún criterio médico profesional.'
)

st.write('Ingresa los 30 valores de la muestra (o usa los valores promedio precargados) para clasificarla.')

mean_feats = feature_names[0:10]
error_feats = feature_names[10:20]
worst_feats = feature_names[20:30]

# Valores por defecto = media del dataset original (recuperada del scaler ya ajustado)
default_values = dict(zip(feature_names, scaler_bc.mean_))

inputs = {}
with st.form('sample_form'):
    tab1, tab2, tab3 = st.tabs(['Valores medios (mean)', 'Error estándar (SE)', 'Peor valor (worst)'])
    with tab1:
        cols = st.columns(2)
        for i, feat in enumerate(mean_feats):
            inputs[feat] = cols[i % 2].number_input(feat, value=float(round(default_values[feat], 4)), format='%.4f')
    with tab2:
        cols = st.columns(2)
        for i, feat in enumerate(error_feats):
            inputs[feat] = cols[i % 2].number_input(feat, value=float(round(default_values[feat], 4)), format='%.4f')
    with tab3:
        cols = st.columns(2)
        for i, feat in enumerate(worst_feats):
            inputs[feat] = cols[i % 2].number_input(feat, value=float(round(default_values[feat], 4)), format='%.4f')
    submitted = st.form_submit_button('Clasificar muestra')

if submitted:
    X_new = np.array([[inputs[f] for f in feature_names]])
    X_new_scaled = scaler_bc.transform(X_new)
    X_new_pca = pca_bc.transform(X_new_scaled)
    cluster_pred = int(knn_bc.predict(X_new_pca)[0])
    profile = cluster_profiles[cluster_pred]

    if 'malignidad' in profile['label'].lower():
        st.error(f'Segmento asignado: {profile["label"]}')
    else:
        st.success(f'Segmento asignado: {profile["label"]}')
    st.info(f'Recomendación: {profile["recomendacion"]}')
    if profile.get('recall_maligno_historico') is not None:
        st.metric('Recall histórico de este segmento sobre casos malignos reales',
                   f'{profile["recall_maligno_historico"] * 100:.1f}%')

st.caption('Caso de estudio canónico (4.7) - Taller 6, ICYA3004, Universidad de los Andes.')
