import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import matplotlib.pyplot as plt # Importado para evitar el error de la captura

st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")

# --- LÓGICA DE LIMPIEZA ---
def clean_data(df):
    for col in ['CANTIDAD', 'PRECIO_UNITARIO', 'VENTA_NETA']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    return df.dropna(subset=['SKU', 'CANTIDAD', 'PRECIO_UNITARIO'])

# --- APP ---
st.title("🚀 Pricing Manager: Creoqueyaquedo Edition")

tab1, tab2, tab3 = st.tabs(["📁 Input", "📊 Analytics", "💰 Simulator"])

with tab1:
    f = st.file_uploader("Cargar Base de Ventas", type=['csv'])
    if f:
        df_raw = pd.read_csv(f)
        df = clean_data(df_raw)
        st.session_state['main_df'] = df
        
        # Semáforo de tu notebook
        per_lost = (len(df_raw)-len(df))/len(df_raw)*100
        if per_lost < 25: st.success(f"🟢 Base Saludable: {per_lost:.1f}% removido")
        else: st.warning(f"🟡 Revisar Datos: {per_lost:.1f}% removido")

with tab2:
    if 'main_df' in st.session_state:
        df = st.session_state['main_df']
        # Lógica de elasticidad del bloque 4 del notebook
        skus = st.multiselect("SKUs a evaluar", df['SKU'].unique())
        df_sel = df[df['SKU'].isin(skus)] if skus else df
        
        fig = px.scatter(df_sel, x='PRECIO_UNITARIO', y='CANTIDAD', color='SKU', trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if 'main_df' in st.session_state:
        st.header("Proyecciones Financieras")
        ajuste = st.slider("Ajuste de Precio (%)", -15, 15, 0) / 100
        
        # Simulación de Ticket Promedio y Margen (Bloque 6 del notebook)
        res = df.groupby('SKU').agg({'CANTIDAD':'sum', 'VENTA_NETA':'sum', 'PRECIO_UNITARIO':'mean'}).reset_index()
        res['Nuevo_Precio'] = res['PRECIO_UNITARIO'] * (1 + ajuste)
        res['Nueva_Venta'] = res['VENTA_NETA'] * (1 + (ajuste * -1.5)) # Elasticidad promedio supuesta
        
        st.dataframe(res)
        st.download_button("Descargar CSV Final", res.to_csv(), "pricing_final.csv")
