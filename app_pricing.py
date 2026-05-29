import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Pricing OfficeMax Pro", layout="wide")

# --- FUNCIONES CORE (LÓGICA NOTEBOOK) ---

def limpiar_datos_officemax(df):
    """Mantiene la lógica del Bloque 2 del notebook con tus columnas reales"""
    # Columnas numéricas según tu CSV
    cols_num = ['qty', 'net_sale', 'utilidad', 'margen']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    
    # Filtro esencial para logaritmos (Bloque 4)
    # qty = Cantidad, net_sale = Precio/Venta
    df = df.dropna(subset=['prod_nbr', 'qty', 'net_sale'])
    return df[(df['qty'] > 0) & (df['net_sale'] > 0)]

def calcular_elasticidad_notebook(df_subset):
    """Lógica del Bloque 4: Regresión OLS Log-Log sobre el subset filtrado"""
    resultados = []
    skus = df_subset['prod_nbr'].unique()
    
    for sku in skus:
        sub = df_subset[df_subset['prod_nbr'] == sku].copy()
        if len(sub) < 5: continue # Requisito estadístico del notebook
        
        try:
            # Transformación Log-Log
            y = np.log(sub['qty'].values.astype(float))
            x = np.log(sub['net_sale'].values.astype(float))
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            
            resultados.append({
                'prod_nbr': sku,
                'Elasticidad': model.params[1],
                'R2': model.rsquared,
                'P_Value': model.pvalues[1],
                'Venta_Acumulada_Qty': sub['qty'].sum(),
                'Precio_Promedio': sub['net_sale'].mean(),
                'Margen_Actual': sub['margen'].mean() if 'margen' in sub.columns else 0
            })
        except: continue
    return pd.DataFrame(resultados)

# --- ESTRUCTURA DE LA APP (SIDEBAR FILTRABLE) ---

st.sidebar.title("🔍 Filtros Granulares")
archivo = st.sidebar.file_uploader("Cargar Venta_OfficeMax_Ticket.csv", type=['csv'])

if archivo:
    # 1. Carga y Limpieza inicial
    df_raw = pd.read_csv(archivo)
    df_clean = limpiar_datos_officemax(df_raw)
    
    # 2. Navegación Granular (Sidebar)
    # Filtro Departamento
    deptos = sorted(df_clean['dept_nm'].unique())
    sel_depto = st.sidebar.multiselect("Departamento", deptos, key="filter_depto")
    
    df_f = df_clean.copy()
    if sel_depto:
        df_f = df_f[df_f['dept_nm'].isin(sel_depto)]
    
    # Filtro Clase (Categoría)
    clases = sorted(df_f['class_nm'].unique())
    sel_clase = st.sidebar.multiselect("Clase / Categoría", clases, key="filter_clase")
    if sel_clase:
        df_f = df_f[df_f['class_nm'].isin(sel_clase)]
        
    # Filtro Marca
    marcas = sorted(df_f['marca'].unique())
    sel_marca = st.sidebar.multiselect("Marca", marcas, key="filter_marca")
    if sel_marca:
        df_f = df_f[df_f['marca'].isin(sel_marca)]

    # --- PESTAÑAS PRINCIPALES ---
    tab1, tab2, tab3 = st.tabs(["📊 Resumen Segmento", "📈 Elasticidad", "💰 Simulador"])

    with tab1:
        st.subheader("Métricas del Segmento Filtrado")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SKUs Seleccionados", df_f['prod_nbr'].nunique())
        c2.metric("Volumen (Qty)", f"{int(df_f['qty'].sum()):,}")
        c3.metric("Ticket Promedio", f"${df_f['net_sale'].mean():.2f}")
        c4.metric("Margen Promedio", f"{df_f['margen'].mean()*100:.1f}%")
        
        st.write("Vista previa de transacciones filtradas:")
        st.dataframe(df_f.head(50), use_container_width=True)

    with tab2:
        st.subheader("Análisis de Demanda (Notebook Bloque 4)")
        df_elast = calcular_elasticidad_notebook(df_f)
        
        if not df_elast.empty:
            col_graph, col_table = st.columns([2, 1])
            
            with col_table:
                st.write("Resultados OLS")
                st.dataframe(df_elast[['prod_nbr', 'Elasticidad', 'R2']].sort_values('Elasticidad'), hide_index=True)
            
            with col_graph:
                sku_sel = st.selectbox("Analizar SKU específico:", df_elast['prod_nbr'].unique())
                sub_viz = df_f[df_f['prod_nbr'] == sku_sel]
                fig = px.scatter(sub_viz, x='net_sale', y='qty', trendline="ols",
                                 title=f"Curva de Demanda: SKU {sku_sel}",
                                 labels={'net_sale': 'Precio Unitario', 'qty': 'Cantidad'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos históricos en este filtro para calcular elasticidad (se requieren 5+ puntos de precio por SKU).")

    with tab3:
        st.subheader("Simulador de Impacto (Notebook Bloque 6)")
        if 'df_elast' in locals() and not df_elast.empty:
            # Controles de Escenario
            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                ajuste_p = st.select_slider("Ajuste de Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15], value=0.0)
            with col_sim2:
                promo = st.selectbox("Mecánica Promocional:", ["Ninguna", "2x1", "3x2", "2do al 50%"])
            
            # Cálculos de Proyección
            df_sim = df_elast.copy()
            df_sim['Nuevo_Precio'] = df_sim['Precio_Promedio'] * (1 + ajuste_p)
            # %ΔQ = %ΔP * Elasticidad
            df_sim['Nueva_Demanda'] = df_sim['Venta_Acumulada_Qty'] * (1 + (df_sim['Elasticidad'] * ajuste_p))
            
            # Multiplicadores promocionales del notebook
            if promo == "2x1": df_sim['Nueva_Demanda'] *= 1.85
            elif promo == "3x2": df_sim['Nueva_Demanda'] *= 1.55
            elif promo == "2do al 50%": df_sim['Nueva_Demanda'] *= 1.30
            
            df_sim['Impacto_Venta_Bruta'] = (df_sim['Nuevo_Precio'] * df_sim['Nueva_Demanda']) - (df_sim['Precio_Promedio'] * df_sim['Venta_Acumulada_Qty'])
            
            st.dataframe(df_sim[['prod_nbr', 'Elasticidad', 'Precio_Promedio', 'Nuevo_Precio', 'Nueva_Demanda', 'Impacto_Venta_Bruta']].style.format(precision=2))
            
            # Dictamen
            avg_e = df_sim['Elasticidad'].mean()
            if avg_e < -1:
                st.success(f"**Estrategia Recomendada:** El segmento es ELÁSTICO ({avg_e:.2f}). Se sugiere traccionar VOLUMEN con la promoción '{promo}'.")
            else:
                st.info(f"**Estrategia Recomendada:** El segmento es INELÁSTICO ({avg_e:.2f}). Se sugiere defender MARGEN ajustando precio al alza.")
        else:
            st.error("Carga datos y aplica filtros para activar la simulación.")

else:
    st.info("👋 Bienvenida. Por favor sube el archivo 'Venta_OfficeMax_Ticket.csv' en el panel izquierdo para iniciar.")
