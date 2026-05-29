import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# Configuración visual estilo Capturas
st.set_page_config(page_title="Pricing Dinámico Pro", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #eee; }
    .main-header { font-size: 24px; font-weight: bold; color: #1E3A8A; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR DE CÁLCULO 'CREOQUEYAQUEDO' ---

def procesar_metricas_notebook(df):
    """Recrea las columnas de pricing del notebook original"""
    # Limpieza de tipos para evitar errores previos
    for col in ['CANTIDAD', 'PRECIO_UNITARIO', 'VENTA_NETA']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    df = df.dropna(subset=['SKU', 'CANTIDAD', 'PRECIO_UNITARIO'])

    # Agregación por SKU (Pricing Base)
    res = df.groupby('SKU').agg({
        'CANTIDAD': 'sum',
        'PRECIO_UNITARIO': 'mean',
        'VENTA_NETA': 'sum'
    }).reset_index()

    res.columns = ['SKU', 'Unidades_Base', 'Precio_Base', 'Ingreso_Base']
    
    # Cálculos de Margen (Lógica Notebook)
    res['Costo_Unitario_Base'] = res['Precio_Base'] * 0.65 # Ajustable según tu industria
    res['Margen_Unitario_Base'] = res['Precio_Base'] - res['Costo_Unitario_Base']
    res['Margen_Base'] = res['Margen_Unitario_Base'] * res['Unidades_Base']
    res['Ticket_Promedio_Linea'] = res['Ingreso_Base'] / res['Unidades_Base'].replace(0, 1)
    
    return res, df

def calcular_elasticidad_full(df_historico):
    """Cálculo estadístico completo: Beta, R2 y P-Value"""
    stats_list = []
    for sku in df_historico['SKU'].unique():
        sub = df_historico[df_historico['SKU'] == sku].copy()
        sub = sub[(sub['CANTIDAD'] > 0) & (sub['PRECIO_UNITARIO'] > 0)]
        
        if len(sub) < 5: continue
        
        try:
            y = np.log(sub['CANTIDAD'].values.astype(float))
            x = np.log(sub['PRECIO_UNITARIO'].values.astype(float))
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            
            stats_list.append({
                'SKU': sku,
                'Elasticidad': model.params[1],
                'R2': model.rsquared,
                'P_Value': model.pvalues[1],
                'Elasticidad_Disponible': True
            })
        except: continue
    return pd.DataFrame(stats_list)

# --- INTERFAZ STREAMLIT ---

tab1, tab2, tab3 = st.tabs(["📊 Diagnóstico de Base", "📈 Análisis de Elasticidad", "💰 Simulador de Pricing"])

with tab1:
    st.markdown("<p class='main-header'>Carga y Calidad de Datos</p>", unsafe_allow_html=True)
    file = st.file_uploader("Subir Ventas CSV", type=['csv'])
    
    if file:
        df_raw = pd.read_csv(file)
        # Procesar con lógica del notebook
        df_pricing_base, df_clean = procesar_metricas_notebook(df_raw)
        st.session_state['df_clean'] = df_clean
        st.session_state['df_base'] = df_pricing_base
        
        # Semáforo
        removidos = len(df_raw) - len(df_clean)
        pct = (removidos / len(df_raw)) * 100
        
        if pct < 25: st.success(f"🟢 VERDE: Base óptima ({pct:.1f}% removido)")
        elif pct < 50: st.warning(f"🟡 AMARILLO: Revisar varianza ({pct:.1f}% removido)")
        else: st.error(f"🔴 ROJO: Base crítica ({pct:.1f}% removido)")
        
        st.dataframe(df_pricing_base.head())

with tab2:
    if 'df_clean' in st.session_state:
        st.markdown("<p class='main-header'>Elasticidad y Demanda</p>", unsafe_allow_html=True)
        df_elast = calcular_elasticidad_full(st.session_state['df_clean'])
        
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.write("Métricas de Regresión")
            st.dataframe(df_elast[['SKU', 'Elasticidad', 'R2', 'P_Value']])
        
        with col_b:
            sku_sel = st.selectbox("Analizar SKU específico", st.session_state['df_clean']['SKU'].unique())
            sub_plot = st.session_state['df_clean'][st.session_state['df_clean']['SKU'] == sku_sel]
            fig = px.scatter(sub_plot, x='PRECIO_UNITARIO', y='CANTIDAD', trendline="ols", title=f"Curva de Demanda: {sku_sel}")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    if 'df_base' in st.session_state and not df_elast.empty:
        st.markdown("<p class='main-header'>Proyecciones de Pricing Dinámico</p>", unsafe_allow_html=True)
        
        # Unir Pricing con Elasticidad
        df_final = st.session_state['df_base'].merge(df_elast, on='SKU', how='left')
        
        c1, c2, c3 = st.columns(3)
        with c1:
            cambio = st.select_slider("Ajuste de Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
        with c2:
            promo = st.selectbox("Promoción", ["Ninguna", "2x1", "3x2", "2do al 50%"])
            
        # Lógica de Proyección
        df_final['Nuevo_Precio'] = df_final['Precio_Base'] * (1 + cambio)
        # Q_n = Q_b * (1 + (e * %cambio))
        df_final['Nueva_Demanda'] = df_final['Unidades_Base'] * (1 + (df_final['Elasticidad'] * cambio))
        
        # Efecto Promoción (Simulado según notebook)
        if promo == "2x1": df_final['Nueva_Demanda'] *= 1.9
        elif promo == "3x2": df_final['Nueva_Demanda'] *= 1.6
        
        df_final['Nuevo_Ingreso'] = df_final['Nuevo_Precio'] * df_final['Nueva_Demanda']
        df_final['Nuevo_Margen'] = (df_final['Nuevo_Precio'] - df_final['Costo_Unitario_Base']) * df_final['Nueva_Demanda']

        st.dataframe(df_final[['SKU', 'Precio_Base', 'Nuevo_Precio', 'Ingreso_Base', 'Nuevo_Ingreso', 'Margen_Base', 'Nuevo_Margen']])
        
        # Dictamen Personalizado (Insight)
        for s in df_final['SKU'].unique()[:1]:
            e_val = df_final[df_final['SKU'] == s]['Elasticidad'].values[0]
            st.info(f"**Dictamen Analítico para SKU {s}:** Demanda {'Muy Elástica' if e_val < -1.2 else 'Poco Elástica'}. Recomendación: {'BAJAR PRECIO' if e_val < -1.2 else 'SUBIR PRECIO'}.")

        st.download_button("Descargar Experimentos Globales", df_final.to_csv(), "pricing_pro.csv")
