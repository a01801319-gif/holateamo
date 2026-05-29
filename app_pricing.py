import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# --- CONFIGURACIÓN TÉCNICA ---
st.set_page_config(page_title="Pricing Manager - Estilo Notebook", layout="wide")

def limpiar_datos_notebook(df):
    """Lógica del Bloque 2 del notebook"""
    cols = ['SKU', 'CANTIDAD', 'PRECIO_UNITARIO', 'VENTA_NETA']
    for c in cols:
        if c in df.columns:
            # Eliminamos basura de texto y convertimos a número
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    
    # Solo filas con datos completos y positivos para el logaritmo
    df = df.dropna(subset=['SKU', 'CANTIDAD', 'PRECIO_UNITARIO'])
    return df[(df['CANTIDAD'] > 0) & (df['PRECIO_UNITARIO'] > 0)]

def ejecutar_modelo_notebook(df):
    """Lógica del Bloque 4: Elasticidad Log-Log"""
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
                'P_Promedio': sub['PRECIO_UNITARIO'].mean()
            })
        except: continue
    return pd.DataFrame(resultados)

# --- ESTRUCTURA VISUAL ---
st.title("📊 Pricing Dinámico: Creoqueyaquedo")
st.markdown("Adaptación completa del notebook original a herramienta interactiva.")

tab1, tab2, tab3 = st.tabs(["📁 Carga de Datos", "📈 Análisis de Demanda", "💰 Simulador de Precios"])

with tab1:
    st.header("1. Importación y Limpieza")
    archivo = st.file_uploader("Sube tu archivo CSV", type=['csv'])
    
    if archivo:
        df_raw = pd.read_csv(archivo)
        df_clean = limpiar_datos_notebook(df_raw)
        
        # PROCESAMIENTO INMEDIATO: Calculamos todo de una vez para evitar NameErrors
        df_elast = ejecutar_modelo_notebook(df_clean)
        
        # Guardamos en la memoria de la sesión
        st.session_state['df_master'] = df_clean
        st.session_state['df_elast'] = df_elast
        
        st.success(f"✅ ¡Todo listo! Se procesaron {len(df_clean)} registros y {len(df_elast)} SKUs.")
        st.dataframe(df_clean.head())

with tab2:
    st.header("2. Curvas de Elasticidad")
    # Verificamos que los datos existan antes de mostrar nada
    if 'df_master' in st.session_state and 'df_elast' in st.session_state:
        df = st.session_state['df_master']
        res_elast = st.session_state['df_elast']
        
        sku_sel = st.selectbox("Selecciona un producto para analizar:", res_elast['SKU'].unique())
        
        # Gráfica interactiva
        sub_plot = df[df['SKU'] == sku_sel]
        fig = px.scatter(sub_plot, x='PRECIO_UNITARIO', y='CANTIDAD', trendline="ols",
                         title=f"Modelo de Demanda para {sku_sel}",
                         labels={'PRECIO_UNITARIO': 'Precio ($)', 'CANTIDAD': 'Unidades'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Métricas Estadísticas")
        st.dataframe(res_elast.style.format(precision=3))
    else:
        st.info("👆 Por favor, carga un archivo en la primera pestaña para ver el análisis.")

with tab3:
    st.header("3. Simulador de Escenarios Financieros")
    if 'df_elast' in st.session_state:
        res_sim = st.session_state['df_elast'].copy()
        
        c1, c2 = st.columns(2)
        with c1:
            ajuste = st.select_slider("Ajuste de Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
        with c2:
            mecanica = st.selectbox("Mecánica Promocional Sugerida:", ["Ninguna", "2x1", "3x2", "2do al 50%"])
        
        # Lógica de impacto del Bloque 6 del notebook
        res_sim['Nuevo_Precio'] = res_sim['P_Promedio'] * (1 + ajuste)
        res_sim['Nueva_Demanda'] = res_sim['Q_Base'] * (1 + (res_sim['Elasticidad'] * ajuste))
        
        # Multiplicadores de volumen por promoción
        if mecanica == "2x1": res_sim['Nueva_Demanda'] *= 1.85
        elif mecanica == "3x2": res_sim['Nueva_Demanda'] *= 1.55
        
        st.subheader("Proyección de Resultados")
        st.dataframe(res_sim[['SKU', 'Elasticidad', 'P_Promedio', 'Nuevo_Precio', 'Nueva_Demanda']])
        
        # Dictámenes del notebook
        st.markdown("### 💡 Insights Estratégicos")
        for i, r in res_sim.head(3).iterrows():
            rec = "🔴 INELÁSTICO: Foco en MARGEN (Subir Precio)" if r['Elasticidad'] > -1 else "🟢 ELÁSTICO: Foco en VOLUMEN (Bajar Precio)"
            st.warning(f"**SKU {r['SKU']}:** {rec}")
    else:
        st.info("👆 Carga tus datos para activar el simulador.")
