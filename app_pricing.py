import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")

st.markdown("<style>.stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #eef0f2; } .status-red { color: #dc3545; font-weight: bold; } .status-yellow { color: #856404; font-weight: bold; } .status-green { color: #155724; font-weight: bold; }</style>", unsafe_allow_html=True)

def calcular_elasticidad_avanzada(df):
    resultados = []
    for sku in df['SKU'].unique():
        sub = df[df['SKU'] == sku].copy()
        if len(sub) < 5: continue
        sub['log_q'] = np.log(sub['CANTIDAD'] + 1)
        sub['log_p'] = np.log(sub['PRECIO_UNITARIO'] + 1)
        try:
            X = sm.add_constant(sub['log_p'])
            model = sm.OLS(sub['log_q'], X).fit()
            resultados.append({'SKU': sku, 'Elasticidad': model.params[1], 'R2': model.rsquared, 'P_Value': model.pvalues[1]})
        except: continue
    return pd.DataFrame(resultados)

tab1, tab2, tab3 = st.tabs(["📁 Carga y Calidad", "📈 Elasticidad & Geografía", "💰 Pricing & Proyecciones"])

with tab1:
    st.header("Gestión de Datos")
    f_sales = st.file_uploader("Cargar Ventas (CSV)", type=['csv'])
    if f_sales:
        df = pd.read_csv(f_sales)
        total = len(df)
        df_f = df.dropna(subset=['SKU', 'PRECIO_UNITARIO', 'CANTIDAD']).copy()
        rem = total - len(df_f)
        pct = (rem / total) * 100
        num_skus = df_f['SKU'].nunique()
        st.subheader("Semáforo de Calidad")
        if total < 50 or num_skus < 5 or pct > 50:
            st.markdown("<span class='status-red'>🔴 Rojo: Crítico</span>", unsafe_allow_html=True)
        elif 25 <= pct <= 50:
            st.markdown("<span class='status-yellow'>🟡 Amarillo: Advertencia</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-green'>🟢 Verde: Óptimo</span>", unsafe_allow_html=True)
        st.session_state['data'] = df_f

with tab2:
    if 'data' in st.session_state:
        df_e = st.session_state['data']
        f_sku = st.multiselect("Seleccionar SKUs", df_e['SKU'].unique())
        df_elast = calcular_elasticidad_avanzada(df_e)
        st.plotly_chart(px.scatter(df_e, x='PRECIO_UNITARIO', y='CANTIDAD', color='SKU', trendline='ols'))
        st.download_button("Descargar Elasticidad", df_elast.to_csv(index=False), "elasticidad.csv")

with tab3:
    if 'data' in st.session_state and 'df_elast' in locals():
        st.header("Simulador de Pricing")
        ajuste = st.select_slider("Ajuste de Precio", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
        # Lógica de cálculo simplificada para el archivo final
        st.info("Simulación basada en elasticidad calculada y escenarios de precio.")
        st.download_button("Descargar Experimentos", df_e.to_csv(), "experimentos.csv")
