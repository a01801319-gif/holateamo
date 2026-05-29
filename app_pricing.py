import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# --- CONFIGURACIÓN ESTILO NOTEBOOK ---
st.set_page_config(page_title="Pricing Manager - CreoQueYaQuedo", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #dce1e6; }
    .status-box { padding: 10px; border-radius: 5px; font-weight: bold; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA EXTRAÍDA DEL NOTEBOOK ---

def limpiar_datos_notebook(df):
    """Lógica del Bloque 2: Limpieza y tipado"""
    cols = ['SKU', 'CANTIDAD', 'PRECIO_UNITARIO', 'VENTA_NETA']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    return df.dropna(subset=['SKU', 'CANTIDAD', 'PRECIO_UNITARIO'])

def calcular_elasticidad_notebook(df):
    """Lógica del Bloque 4: Regresión OLS Log-Log"""
    resultados = []
    for sku in df['SKU'].unique():
        sub = df[(df['SKU'] == sku) & (df['CANTIDAD'] > 0) & (df['PRECIO_UNITARIO'] > 0)].copy()
        if len(sub) < 5: continue
        try:
            # Uso de numpy nativo para evitar NameError: plt o similares
            y = np.log(sub['CANTIDAD'].values.astype(float))
            x = np.log(sub['PRECIO_UNITARIO'].values.astype(float))
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            
            resultados.append({
                'SKU': sku,
                'Elasticidad': model.params[1],
                'R2': model.rsquared,
                'P_Value': model.pvalues[1],
                'Q_Promedio': sub['CANTIDAD'].mean(),
                'P_Promedio': sub['PRECIO_UNITARIO'].mean()
            })
        except: continue
    return pd.DataFrame(resultados)

# --- ESTRUCTURA APP ---

st.title("📊 Simulador de Pricing Dinámico")
st.info("Basado en la lógica financiera del notebook: CreoQueYaQuedo.ipynb")

tab1, tab2, tab3 = st.tabs(["📁 Carga de Datos", "📈 Análisis Estadístico", "💰 Simulador de Impacto"])

with tab1:
    st.header("1. Diagnóstico de Base")
    file = st.file_uploader("Subir Archivo CSV", type=['csv'])
    if file:
        df_raw = pd.read_csv(file)
        df_clean = limpiar_datos_notebook(df_raw)
        st.session_state['data'] = df_clean
        
        # Semáforo de Calidad (Bloque 3 Notebook)
        perdida = ((len(df_raw)-len(df_clean))/len(df_raw))*100
        if perdida < 25:
            st.markdown("<div class='status-box' style='background:#d4edda; color:#155724;'>🟢 CALIDAD ALTA: Datos listos</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-box' style='background:#fff3cd; color:#856404;'>🟡 CALIDAD MEDIA: Revisar nulos</div>", unsafe_allow_html=True)
        
        st.write("Vista previa de datos limpios:", df_clean.head())

with tab2:
    if 'data' in st.session_state:
        st.header("2. Resultados de Elasticidad")
        df_elast = calcular_elasticidad_notebook(st.session_state['data'])
        st.session_state['elasticidad'] = df_elast
        
        # Gráfica de dispersión interactiva
        sku_sel = st.selectbox("Seleccionar SKU", df_elast['SKU'].unique())
        sub_plot = st.session_state['data'][st.session_state['data']['SKU'] == sku_sel]
        
        fig = px.scatter(sub_plot, x='PRECIO_UNITARIO', y='CANTIDAD', trendline="ols", 
                         title=f"Regresión Log-Log: {sku_sel}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_elast.style.format(precision=3))

with tab3:
    if 'elasticidad' in st.session_state:
        st.header("3. Escenarios (Bloque 6)")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            ajuste = st.select_slider("Ajuste de Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
        with col_c2:
            mecanica = st.selectbox("Mecánica Promocional", ["Ninguna", "2x1", "3x2", "2do al 50%"])
            
        # Cálculo de Impacto
        df_res = st.session_state['elasticidad'].copy()
        df_res['Nuevo_Precio'] = df_res['P_Promedio'] * (1 + ajuste)
        
        # Q_nueva = Q_base * (1 + (E * %P))
        df_res['Nueva_Demanda'] = df_res['Q_Promedio'] * (1 + (df_res['Elasticidad'] * ajuste))
        
        # Efecto Promoción
        if mecanica == "2x1": df_res['Nueva_Demanda'] *= 1.8
        elif mecanica == "3x2": df_res['Nueva_Demanda'] *= 1.5
        
        st.subheader("Proyección de Resultados")
        st.dataframe(df_res[['SKU', 'Elasticidad', 'P_Promedio', 'Nuevo_Precio', 'Nueva_Demanda']])
        
        # Dictamen (Insight final del notebook)
        for i, r in df_res.head(3).iterrows():
            dictamen = "🔴 INELÁSTICO: Subir precio para margen" if r['Elasticidad'] > -1 else "🟢 ELÁSTICO: Bajar precio para volumen"
            st.success(f"**SKU {r['SKU']}:** {dictamen}")

else:
    st.warning("Esperando carga de archivo en la pestaña 'Carga de Datos'...")
