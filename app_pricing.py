import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Pricing OfficeMax - CreoQueYaQuedo", layout="wide")

# --- MAPEO DE COLUMNAS (Basado en tu CSV de OfficeMax) ---
COL_MAP = {
    'prod_nbr': 'SKU',
    'qty': 'CANTIDAD',
    'net_sale': 'PRECIO_UNITARIO', # En tu CSV parece ser el valor de venta
    'margen': 'MARGEN_PCT'
}

# --- FUNCIONES DEL NOTEBOOK ---

def limpiar_datos_officemax(df):
    """Lógica de limpieza del Bloque 2 adaptada a OfficeMax"""
    # Renombrar para que el código del notebook funcione
    df = df.rename(columns=COL_MAP)
    
    cols_validar = ['SKU', 'CANTIDAD', 'PRECIO_UNITARIO']
    for col in cols_validar:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    
    # Filtro de seguridad (Bloque 4: Logaritmos no aceptan 0 o negativos)
    df = df.dropna(subset=cols_validar)
    df = df[(df['CANTIDAD'] > 0) & (df['PRECIO_UNITARIO'] > 0)]
    return df

def calcular_elasticidad_notebook(df):
    """Lógica del Bloque 4: Regresión OLS Log-Log"""
    resultados = []
    # Agrupamos por SKU para tener puntos históricos de precio/cantidad
    for sku in df['SKU'].unique():
        sub = df[df['SKU'] == sku].copy()
        if len(sub) < 5: continue # Mínimo de datos para validez
        
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
                'Venta_Total': sub['CANTIDAD'].sum(),
                'Precio_Promedio': sub['PRECIO_UNITARIO'].mean(),
                'Margen_Promedio': sub['MARGEN_PCT'].mean() if 'MARGEN_PCT' in sub.columns else 0
            })
        except: continue
    return pd.DataFrame(resultados)

# --- INTERFAZ STREAMLIT ---

st.title("🚀 OfficeMax Pricing Dashboard")
st.markdown("Basado íntegramente en la lógica de `CreoQueYaQuedo.ipynb` aplicado a tus datos reales.")

tab1, tab2, tab3 = st.tabs(["📥 Carga de Archivo", "📈 Análisis de Elasticidad", "💰 Simulador de Escenarios"])

with tab1:
    st.header("Carga de Datos OfficeMax")
    archivo = st.file_uploader("Sube el archivo Venta_OfficeMax_Ticket.csv", type=['csv'])
    
    if archivo:
        df_raw = pd.read_csv(archivo)
        df_clean = limpiar_datos_officemax(df_raw)
        
        # Procesamiento inmediato para evitar errores de variables no definidas
        df_elast = calcular_elasticidad_notebook(df_clean)
        
        st.session_state['df_master'] = df_clean
        st.session_state['df_elast'] = df_elast
        
        st.success(f"✅ Datos cargados y mapeados. {len(df_clean)} registros procesados.")
        st.write("Columnas originales detectadas:", list(df_raw.columns))
        st.dataframe(df_clean.head())

with tab2:
    st.header("Análisis de Demanda (Bloque 4)")
    if 'df_elast' in st.session_state:
        res = st.session_state['df_elast']
        master = st.session_state['df_master']
        
        sku_sel = st.selectbox("Selecciona un Producto (prod_nbr):", res['SKU'].unique())
        
        # Gráfica interactiva de la regresión
        sub_plot = master[master['SKU'] == sku_sel]
        fig = px.scatter(sub_plot, x='PRECIO_UNITARIO', y='CANTIDAD', trendline="ols",
                         title=f"Curva de Demanda Log-Log: {sku_sel}",
                         labels={'PRECIO_UNITARIO': 'Precio Unitario', 'CANTIDAD': 'Cantidad Vendida'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Métricas de Elasticidad")
        st.dataframe(res.style.format(precision=3))
    else:
        st.info("👆 Por favor, carga el CSV en la primera pestaña.")

with tab3:
    st.header("Simulador de Impacto Financiero (Bloque 6)")
    if 'df_elast' in st.session_state:
        df_sim = st.session_state['df_elast'].copy()
        
        c1, c2 = st.columns(2)
        with c1:
            ajuste = st.select_slider("Ajuste de Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
        with c2:
            promo = st.selectbox("Efecto de Promoción Teórica:", ["Ninguna", "2x1", "3x2", "2do al 50%"])
            
        # Lógica de proyección del notebook
        df_sim['Nuevo_Precio'] = df_sim['Precio_Promedio'] * (1 + ajuste)
        # Delta Q = Delta P * Elasticidad
        df_sim['Nueva_Demanda'] = df_sim['Venta_Total'] * (1 + (df_sim['Elasticidad'] * ajuste))
        
        # Factores promocionales del notebook
        if promo == "2x1": df_sim['Nueva_Demanda'] *= 1.85
        elif promo == "3x2": df_sim['Nueva_Demanda'] *= 1.55
        
        st.subheader("Proyección de Ventas")
        st.dataframe(df_sim[['SKU', 'Elasticidad', 'Precio_Promedio', 'Nuevo_Precio', 'Nueva_Demanda']])
        
        # Dictámenes estratégicos
        for _, row in df_sim.head(3).iterrows():
            if row['Elasticidad'] < -1.1:
                st.success(f"**SKU {row['SKU']}**: Altamente Elástico ({row['Elasticidad']:.2f}). Se recomienda BAJAR precio para ganar volumen.")
            else:
                st.warning(f"**SKU {row['SKU']}**: Poco Elástico ({row['Elasticidad']:.2f}). Se recomienda SUBIR precio para maximizar margen.")
    else:
        st.error("❌ No se han calculado elasticidades aún.")
