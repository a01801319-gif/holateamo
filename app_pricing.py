import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Pricing OfficeMax - Final", layout="wide")

# --- FUNCIONES CORE (LÓGICA DEL NOTEBOOK) ---

def limpiar_datos_officemax(df):
    # Forzamos nombres de columnas a minúsculas para evitar errores de escritura
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Columnas numéricas según tu CSV: qty, net_sale, margen
    cols_num = ['qty', 'net_sale', 'margen']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    
    # Filtro para logaritmos (Bloque 4 del notebook)
    df = df.dropna(subset=['prod_nbr', 'qty', 'net_sale'])
    return df[(df['qty'] > 0) & (df['net_sale'] > 0)]

def calcular_elasticidad_notebook(df_subset):
    resultados = []
    # Usamos prod_nbr como identificador
    for sku in df_subset['prod_nbr'].unique():
        sub = df_subset[df_subset['prod_nbr'] == sku]
        if len(sub) < 5: continue
        
        try:
            y = np.log(sub['qty'].values.astype(float))
            x = np.log(sub['net_sale'].values.astype(float))
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            
            resultados.append({
                'prod_nbr': sku,
                'Elasticidad': model.params[1],
                'R2': model.rsquared,
                'Venta_Acumulada': sub['qty'].sum(),
                'Precio_Prom_Historico': sub['net_sale'].mean()
            })
        except: continue
    return pd.DataFrame(resultados)

# --- INTERFAZ ---

st.sidebar.title("📊 Control OfficeMax")
archivo = st.sidebar.file_uploader("Cargar Venta_OfficeMax_Ticket.csv", type=['csv'])

if archivo:
    # 1. Procesamiento Inicial
    if 'df_master' not in st.session_state:
        df_raw = pd.read_csv(archivo)
        st.session_state['df_master'] = limpiar_datos_officemax(df_raw)

    df_main = st.session_state['df_master']

    # 2. Sidebar con Keys únicas para evitar DuplicateWidgetID
    st.sidebar.markdown("---")
    
    # Filtro Departamento (dept_nm)
    depto_list = sorted(df_main['dept_nm'].unique().tolist())
    sel_depto = st.sidebar.multiselect("Departamento", depto_list, key="key_depto")
    
    df_f = df_main.copy()
    if sel_depto:
        df_f = df_f[df_f['dept_nm'].isin(sel_depto)]
    
    # Filtro Categoría (class_nm)
    clase_list = sorted(df_f['class_nm'].unique().tolist())
    sel_clase = st.sidebar.multiselect("Categoría / Clase", clase_list, key="key_clase")
    if sel_clase:
        df_f = df_f[df_f['class_nm'].isin(sel_clase)]

    # --- PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs(["📋 Dashboard", "📈 Análisis Demanda", "💰 Simulador"])

    with tab1:
        st.subheader("Resumen de Segmento")
        col1, col2, col3 = st.columns(3)
        col1.metric("SKUs", df_f['prod_nbr'].nunique())
        col2.metric("Volumen Qty", f"{int(df_f['qty'].sum()):,}")
        col3.metric("Precio Prom.", f"${df_f['net_sale'].mean():.2f}")
        
        st.dataframe(df_f.head(100), use_container_width=True)

    with tab2:
        st.subheader("Elasticidad (Bloque 4)")
        df_elast = calcular_elasticidad_notebook(df_f)
        
        if not df_elast.empty:
            sku_viz = st.selectbox("Seleccionar Producto:", df_elast['prod_nbr'].unique(), key="key_sku_sel")
            fig = px.scatter(df_f[df_f['prod_nbr'] == sku_viz], 
                             x='net_sale', y='qty', trendline="ols",
                             title=f"Modelo Log-Log para {sku_viz}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_elast.sort_values('Elasticidad'))
        else:
            st.warning("Selecciona un segmento con más datos históricos.")

    with tab3:
        if 'df_elast' in locals() and not df_elast.empty:
            st.subheader("Simulador de Escenarios (Bloque 6)")
            ajuste = st.select_slider("Ajuste de Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15], value=0.0)
            promo = st.selectbox("Efecto Promo:", ["Ninguna", "2x1", "3x2"], key="key_promo")
            
            df_sim = df_elast.copy()
            df_sim['Nuevo_Precio'] = df_sim['Precio_Prom_Historico'] * (1 + ajuste)
            df_sim['Nueva_Demanda'] = df_sim['Venta_Acumulada'] * (1 + (df_sim['Elasticidad'] * ajuste))
            
            if promo == "2x1": df_sim['Nueva_Demanda'] *= 1.85
            elif promo == "3x2": df_sim['Nueva_Demanda'] *= 1.55
            
            st.dataframe(df_sim[['prod_nbr', 'Elasticidad', 'Precio_Prom_Historico', 'Nuevo_Precio', 'Nueva_Demanda']])
        else:
            st.info("Carga datos para activar simulación.")

else:
    st.info("👋 Sube el archivo 'Venta_OfficeMax_Ticket.csv' para comenzar.")
