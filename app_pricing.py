import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")

# --- LIMPIEZA DE DATOS ---
def sanitizar_dataframe(df):
    for col in ['CANTIDAD', 'PRECIO_UNITARIO', 'VENTA_NETA']:
        if col in df.columns:
            # Quitamos todo lo que no sea número o punto decimal
            df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['SKU', 'CANTIDAD', 'PRECIO_UNITARIO'])

def calcular_elasticidad_final(df):
    resultados = []
    for sku in df['SKU'].unique():
        sub = df[df['SKU'] == sku].copy()
        # Solo positivos para evitar errores de logaritmo
        sub = sub[(sub['CANTIDAD'] > 0) & (sub['PRECIO_UNITARIO'] > 0)]
        
        if len(sub) < 5: continue
        
        try:
            # PASO CLAVE: Convertimos a arrays de numpy tipo float puro
            y = np.log(sub['CANTIDAD'].values.astype(np.float64))
            x = np.log(sub['PRECIO_UNITARIO'].values.astype(np.float64))
            
            # Aseguramos que x sea una matriz columna para statsmodels
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            
            resultados.append({
                'SKU': sku,
                'Elasticidad': float(model.params[1]), # Forzamos a float de Python
                'R2': float(model.rsquared),
                'P_Value': float(model.pvalues[1])
            })
        except:
            continue
    return pd.DataFrame(resultados)

# --- APP ---
tab1, tab2, tab3 = st.tabs(["📁 Datos", "📈 Elasticidad", "💰 Pricing"])

with tab1:
    st.header("Carga de Ventas")
    archivo = st.file_uploader("Subir base de ventas (CSV)", type=['csv'])
    if archivo:
        df_raw = pd.read_csv(archivo)
        df = sanitizar_dataframe(df_raw)
        st.session_state['data'] = df
        st.success(f"Base cargada: {len(df)} registros válidos.")
        
        # Semáforo de Calidad
        pct_eliminados = ((len(df_raw) - len(df)) / len(df_raw)) * 100
        if pct_eliminados > 50:
            st.error(f"🔴 Calidad Baja: Se eliminó el {pct_eliminados:.1f}% de los datos.")
        elif pct_eliminados > 25:
            st.warning(f"🟡 Calidad Media: Se eliminó el {pct_eliminados:.1f}% de los datos.")
        else:
            st.success(f"🟢 Calidad Alta: Solo se eliminó el {pct_eliminados:.1f}%.")

with tab2:
    if 'data' in st.session_state:
        df = st.session_state['data']
        skus = st.multiselect("Seleccionar SKUs", df['SKU'].unique())
        
        # Gráfica robusta
        df_plot = df[df['SKU'].isin(skus)] if skus else df
        try:
            fig = px.scatter(df_plot, x='PRECIO_UNITARIO', y='CANTIDAD', color='SKU', trendline="ols")
        except:
            fig = px.scatter(df_plot, x='PRECIO_UNITARIO', y='CANTIDAD', color='SKU')
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de Elasticidad
        df_elast = calcular_elasticidad_final(df)
        st.subheader("Resultados de Elasticidad por SKU")
        st.dataframe(df_elast.style.format(precision=4))
        st.session_state['elasticidad'] = df_elast

with tab3:
    if 'data' in st.session_state and 'elasticidad' in st.session_state:
        st.header("Simulador de Pricing")
        ajuste = st.select_slider("Cambiar Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
        
        df_res = st.session_state['elasticidad'].copy()
        # Lógica de impacto simple
        df_res['Impacto_Venta'] = df_res['Elasticidad'] * ajuste
        st.dataframe(df_res)
        
        st.download_button("Descargar Resultados Completos", df_res.to_csv(index=False), "pricing_final.csv")
