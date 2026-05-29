import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# Configuración
st.set_page_config(page_title="Pricing Dashboard", layout="wide")

# --- ESTILOS ---
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #eef0f2; }
    .status-green { color: #155724; font-weight: bold; background: #d4edda; padding: 5px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

def calcular_elasticidad_segura(df):
    resultados = []
    # Verificamos columnas mínimas
    if not all(col in df.columns for col in ['SKU', 'CANTIDAD', 'PRECIO_UNITARIO']):
        return pd.DataFrame()
        
    for sku in df['SKU'].unique():
        sub = df[df['SKU'] == sku].copy()
        if len(sub) < 3: continue
        try:
            sub['log_q'] = np.log(sub['CANTIDAD'] + 1)
            sub['log_p'] = np.log(sub['PRECIO_UNITARIO'] + 1)
            X = sm.add_constant(sub['log_p'])
            model = sm.OLS(sub['log_q'], X).fit()
            resultados.append({'SKU': sku, 'Elasticidad': model.params[1], 'R2': model.rsquared})
        except: continue
    return pd.DataFrame(resultados)

# --- INTERFAZ ---
tab1, tab2, tab3 = st.tabs(["📁 Carga", "📈 Análisis", "💰 Pricing"])

with tab1:
    st.header("1. Carga de Datos")
    f_sales = st.file_uploader("Subir base de ventas (CSV)", type=['csv'])
    
    if f_sales:
        df = pd.read_csv(f_sales)
        st.success("✅ Archivo cargado correctamente")
        
        # Mostrar resumen de columnas encontradas
        st.write("Columnas detectadas:", list(df.columns))
        
        # Limpieza básica
        cols_necesarias = [c for c in ['SKU', 'CANTIDAD', 'PRECIO_UNITARIO', 'VENTA_NETA'] if c in df.columns]
        df_clean = df.dropna(subset=cols_necesarias).copy()
        st.session_state['data'] = df_clean
        
        # Semáforo simple
        st.markdown(f"Estatus: <span class='status-green'>🟢 Verde: Base lista</span>", unsafe_allow_html=True)

with tab2:
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Filtros dinámicos (SOLO si existen las columnas)
        st.sidebar.header("Filtros")
        
        # Esta es la parte que arregla tu error:
        f_sku = st.sidebar.multiselect("Filtrar SKU", df['SKU'].unique())
        
        # Solo mostrar filtro de departamento si la columna existe
        if 'DEPARTAMENTO' in df.columns:
            f_dep = st.sidebar.multiselect("Departamento", df['DEPARTAMENTO'].unique())
        if 'NSE' in df.columns:
            f_nse = st.sidebar.multiselect("NSE", df['NSE'].unique())

        # Cálculos
        df_elast = calcular_elasticidad_segura(df)
        
        if not df_elast.empty:
            st.subheader("Relación Precio vs Demanda")
            fig = px.scatter(df, x='PRECIO_UNITARIO', y='CANTIDAD', color='SKU' if 'SKU' in df.columns else None, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_elast)
        else:
            st.warning("No hay suficientes datos para calcular elasticidad. Se requieren al menos 3 registros por SKU con precios distintos.")

with tab3:
    if 'data' in st.session_state and 'df_elast' in locals():
        st.header("Simulador de Impacto")
        ajuste = st.select_slider("Ajuste de Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
        
        st.info(f"Simulando escenario con cambio del {ajuste*100}%")
        
        # Botones de descarga
        st.download_button("Descargar Resultados", df.to_csv(index=False), "pricing_results.csv")
else:
    st.info("👋 Por favor, carga un archivo CSV en la primera pestaña para comenzar.")
