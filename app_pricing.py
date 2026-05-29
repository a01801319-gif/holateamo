import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")

# --- FUNCIONES DE LIMPIEZA ---
def limpiar_numeros(df, columnas):
    for col in columnas:
        if col in df.columns:
            # Convertimos a string, quitamos comas/signos y pasamos a numérico
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
    return df

def calcular_elasticidad_pro(df):
    resultados = []
    if not all(col in df.columns for col in ['SKU', 'CANTIDAD', 'PRECIO_UNITARIO']):
        return pd.DataFrame()

    for sku in df['SKU'].unique():
        sub = df[df['SKU'] == sku].copy()
        sub = sub[(sub['CANTIDAD'] > 0) & (sub['PRECIO_UNITARIO'] > 0)].dropna()
        
        if len(sub) < 5: continue
        
        try:
            sub['log_q'] = np.log(sub['CANTIDAD'])
            sub['log_p'] = np.log(sub['PRECIO_UNITARIO'])
            X = sm.add_constant(sub['log_p'])
            model = sm.OLS(sub['log_q'], X).fit()
            
            resultados.append({
                'SKU': sku, 
                'Elasticidad': model.params[1], 
                'R2': model.rsquared,
                'P_Value': model.pvalues[1]
            })
        except: continue
    return pd.DataFrame(resultados)

# --- INTERFAZ ---
tab1, tab2, tab3 = st.tabs(["📁 Datos", "📈 Elasticidad", "💰 Pricing"])

with tab1:
    st.header("Carga de Ventas")
    f = st.file_uploader("Subir CSV", type=['csv'])
    if f:
        df_raw = pd.read_csv(f)
        # LIMPIEZA CRÍTICA: Forzamos números para evitar el error de la captura
        df = limpiar_numeros(df_raw, ['CANTIDAD', 'PRECIO_UNITARIO', 'VENTA_NETA'])
        df = df.dropna(subset=['SKU', 'CANTIDAD', 'PRECIO_UNITARIO'])
        
        st.session_state['df'] = df
        st.success(f"Base lista con {len(df)} registros numéricos.")
        st.write("Tipos de datos:", df[['CANTIDAD', 'PRECIO_UNITARIO']].dtypes)

with tab2:
    if 'df' in st.session_state:
        df = st.session_state['df']
        skus = st.multiselect("SKUs", df['SKU'].unique())
        df_plot = df[df['SKU'].isin(skus)] if skus else df
        
        st.subheader("Relación Demanda vs Precio")
        # El try/except evita que la app truene si la tendencia falla
        try:
            fig = px.scatter(df_plot, x='PRECIO_UNITARIO', y='CANTIDAD', color='SKU', trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo trazar la tendencia: {e}")
            st.plotly_chart(px.scatter(df_plot, x='PRECIO_UNITARIO', y='CANTIDAD', color='SKU'))

        df_res = calcular_elasticidad_pro(df)
        st.dataframe(df_res)

with tab3:
    if 'df' in st.session_state:
        st.header("Simulador de Pricing")
        ajuste = st.select_slider("Cambio Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
        st.write(f"Escenario: {ajuste*100}%")
        st.download_button("Descargar CSV", df.to_csv(index=False), "pricing.csv")
