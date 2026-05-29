import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA (ESTILO DASHBOARD) ---
st.set_page_config(page_title="Pricing Dinámico - Basado en Notebook", layout="wide")

st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #dce1e6; }
    .dictamen-box { padding: 20px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #1E3A8A; background: #eef2ff; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES CORE DEL NOTEBOOK ---

def limpiar_datos_notebook(df):
    """Recrea la lógica del Bloque 2 del notebook"""
    columnas_clave = ['SKU', 'CANTIDAD', 'PRECIO_UNITARIO', 'VENTA_NETA']
    for col in columnas_clave:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    
    # Eliminar nulos en columnas vitales
    df = df.dropna(subset=['SKU', 'CANTIDAD', 'PRECIO_UNITARIO'])
    # Filtrar solo valores positivos (indispensable para logaritmos)
    df = df[(df['CANTIDAD'] > 0) & (df['PRECIO_UNITARIO'] > 0)]
    return df

def calcular_elasticidad_notebook(df):
    """Lógica exacta del Bloque 4: Regresión Log-Log"""
    resultados = []
    skus = df['SKU'].unique()
    
    for sku in skus:
        sub = df[df['SKU'] == sku].copy()
        if len(sub) < 5: continue # Mínimo de datos para validez estadística
        
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
                'Unidades_Historicas': sub['CANTIDAD'].sum(),
                'Precio_Promedio': sub['PRECIO_UNITARIO'].mean()
            })
        except: continue
    return pd.DataFrame(resultados)

# --- INTERFAZ DE USUARIO ---

st.title("📊 Herramienta de Optimización de Pricing")
st.caption("Versión desplegable basada íntegramente en el análisis 'CreoQueYaQuedo.ipynb'")

tab1, tab2, tab3 = st.tabs(["📥 Carga de Datos", "🔬 Análisis Estadístico", "💰 Simulador de Impacto"])

with tab1:
    st.header("1. Importación de Datos")
    uploaded_file = st.file_uploader("Sube tu archivo de ventas (CSV)", type=['csv'])
    
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        df_clean = limpiar_datos_notebook(df_raw)
        st.session_state['data'] = df_clean
        
        # Resumen de calidad (Semáforo del notebook)
        col1, col2, col3 = st.columns(3)
        perdida = ((len(df_raw)-len(df_clean))/len(df_raw))*100
        
        with col1:
            st.metric("Registros Procesados", len(df_clean))
        with col2:
            st.metric("SKUs Detectados", df_clean['SKU'].nunique())
        with col3:
            status = "🟢 Óptimo" if perdida < 20 else "🟡 Revisar" if perdida < 50 else "🔴 Crítico"
            st.metric("Estatus de Calidad", status, f"-{perdida:.1f}% datos")

with tab2:
    if 'data' in st.session_state:
        st.header("2. Resultados de Elasticidad")
        df_e = calcular_elasticidad_notebook(st.session_state['data'])
        st.session_state['elasticidad_df'] = df_e
        
        # Gráfica interactiva de apoyo
        sku_to_plot = st.selectbox("Selecciona un SKU para ver su curva de demanda:", df_e['SKU'].unique())
        sub_plot = st.session_state['data'][st.session_state['data']['SKU'] == sku_to_plot]
        
        fig = px.scatter(sub_plot, x='PRECIO_UNITARIO', y='CANTIDAD', trendline="ols",
                         title=f"Regresión Log-Log para {sku_to_plot}",
                         labels={'PRECIO_UNITARIO': 'Precio ($)', 'CANTIDAD': 'Unidades Vendidas'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_e.style.format(precision=3), use_container_width=True)

with tab3:
    if 'elasticidad_df' in st.session_state:
        st.header("3. Simulador de Escenarios (Bloque 6)")
        
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            porcentaje_cambio = st.select_slider("Ajuste de Precio Sugerido (%)", 
                                                options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15],
                                                value=0.0)
        with col_input2:
            promo_extra = st.selectbox("Aplicar Mecánica Promocional:", 
                                      ["Ninguna", "2x1", "3x2", "50% en 2da unidad"])

        # Lógica de cálculo de impacto
        df_sim = st.session_state['elasticidad_df'].copy()
        
        # Impacto por elasticidad: % Q = % P * Elasticidad
        df_sim['Cambio_Demanda_Pct'] = porcentaje_cambio * df_sim['Elasticidad']
        
        # Factor promocional (multiplicador de volumen según notebook)
        factor_promo = 1.0
        if promo_extra == "2x1": factor_promo = 1.85
        elif promo_extra == "3x2": factor_promo = 1.50
        elif promo_extra == "50% en 2da unidad": factor_promo = 1.30
        
        df_sim['Nueva_Demanda_Unidades'] = df_sim['Unidades_Historicas'] * (1 + df_sim['Cambio_Demanda_Pct']) * factor_promo
        df_sim['Nuevo_Precio'] = df_sim['Precio_Promedio'] * (1 + porcentaje_cambio)
        
        # Mostrar resultados de impacto
        st.subheader("Resultados de la Simulación")
        st.dataframe(df_sim[['SKU', 'Elasticidad', 'Precio_Promedio', 'Nuevo_Precio', 'Nueva_Demanda_Unidades']])

        # Dictamen LLM (Bloque final del notebook)
        st.markdown("### 💡 Dictamen Estratégico")
        for index, row in df_sim.head(3).iterrows():
            rec = "BAJAR PRECIO" if row['Elasticidad'] < -1.2 else "SUBIR PRECIO / OPTIMIZAR"
            st.markdown(f"""
            <div class="dictamen-box">
                <b>SKU: {row['SKU']}</b><br>
                Elasticidad detectada de <b>{row['Elasticidad']:.2f}</b>. <br>
                Estrategia recomendada: <b>{rec}</b>
            </div>
            """, unsafe_allow_html=True)
            
        st.download_button("Exportar Simulación a CSV", df_sim.to_csv(index=False), "simulacion_pricing.csv")

else:
    st.info("👋 Por favor, carga tu base de datos en la primera pestaña para activar el cerebro del notebook.")
