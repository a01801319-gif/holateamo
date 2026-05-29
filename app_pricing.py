import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Pricing Pro - CreoQueYaQuedo", layout="wide")

# --- FUNCIONES CORE DEL NOTEBOOK ---
def limpiar_datos_estricto(df):
    # Lógica del Bloque 2 del notebook
    cols_numericas = ['CANTIDAD', 'PRECIO_UNITARIO', 'VENTA_NETA']
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    
    # Filtro de seguridad para logaritmos (Bloque 4)
    df = df.dropna(subset=['SKU', 'CANTIDAD', 'PRECIO_UNITARIO'])
    df = df[(df['CANTIDAD'] > 0) & (df['PRECIO_UNITARIO'] > 0)]
    return df

def calcular_elasticidad_notebook(df):
    # Lógica del Bloque 4: Regresión OLS Log-Log
    resultados = []
    for sku in df['SKU'].unique():
        sub = df[df['SKU'] == sku].copy()
        if len(sub) < 5: continue
        try:
            y = np.log(sub['CANTIDAD'].values.astype(float))
            x = np.log(sub['PRECIO_UNITARIO'].values.astype(float))
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            
            resultados.append({
                'SKU': sku,
                'Elasticidad': model.params[1],
                'R2': model.rsquared,
                'P_Value': model.pvalues[1],
                'Q_Base': sub['CANTIDAD'].sum(),
                'P_Base': sub['PRECIO_UNITARIO'].mean()
            })
        except: continue
    return pd.DataFrame(resultados)

# --- INTERFAZ ---
st.title("🚀 Dashboard de Pricing Dinámico")
st.markdown("Basado íntegramente en `CreoQueYaQuedo.ipynb`")

tab1, tab2, tab3 = st.tabs(["📥 Carga de Datos", "📈 Análisis Estadístico", "💰 Simulador"])

with tab1:
    st.header("Carga de Ventas")
    archivo = st.file_uploader("Subir CSV", type=['csv'])
    
    if archivo:
        df_raw = pd.read_csv(archivo)
        df_clean = limpiar_datos_estricto(df_raw)
        
        # GUARDAR EN SESSION STATE (Esto evita el NameError)
        st.session_state['df_master'] = df_clean
        
        st.success(f"✅ Datos cargados: {len(df_clean)} registros.")
        st.dataframe(df_clean.head())

with tab2:
    st.header("Análisis de Elasticidad")
    # Verificamos si los datos existen en la "memoria"
    if 'df_master' in st.session_state:
        df = st.session_state['df_master']
        df_elast = calcular_elasticidad_notebook(df)
        st.session_state['df_elast'] = df_elast # Guardamos resultados
        
        sku_sel = st.selectbox("Seleccionar SKU para visualizar curva:", df_elast['SKU'].unique())
        sub_plot = df[df['SKU'] == sku_sel]
        
        fig = px.scatter(sub_plot, x='PRECIO_UNITARIO', y='CANTIDAD', trendline="ols",
                         title=f"Regresión Log-Log: {sku_sel}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_elast.style.format(precision=3))
    else:
        st.warning("⚠️ Primero carga un archivo en la pestaña 'Carga de Datos'.")

with tab3:
    st.header("Simulador de Impacto (Bloque 6)")
    if 'df_elast' in st.session_state:
        df_res = st.session_state['df_elast'].copy()
        
        col1, col2 = st.columns(2)
        with col1:
            ajuste = st.select_slider("Ajuste de Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
        with col2:
            promo = st.selectbox("Efecto Promocional", ["Ninguna", "2x1", "3x2", "2do al 50%"])
        
        # Lógica de proyección del notebook
        df_res['Nuevo_Precio'] = df_res['P_Base'] * (1 + ajuste)
        df_res['Nueva_Demanda'] = df_res['Q_Base'] * (1 + (df_res['Elasticidad'] * ajuste))
        
        # Multiplicadores promocionales (Bloque 6)
        if promo == "2x1": df_res['Nueva_Demanda'] *= 1.85
        elif promo == "3x2": df_res['Nueva_Demanda'] *= 1.50
        
        st.subheader("Resultados de Simulación")
        st.dataframe(df_res[['SKU', 'Elasticidad', 'P_Base', 'Nuevo_Precio', 'Nueva_Demanda']])
        
        # Dictamen Estratégico
        for _, row in df_res.head(3).iterrows():
            rec = "BAJAR PRECIO" if row['Elasticidad'] < -1.2 else "SUBIR PRECIO"
            st.info(f"**Estrategia para {row['SKU']}:** Elasticidad {row['Elasticidad']:.2f} -> **{rec}**")
    else:
        st.error("❌ No hay cálculos de elasticidad disponibles. Revisa la pestaña anterior.")
