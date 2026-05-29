import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Pricing Granular - OfficeMax", layout="wide")

# Estilos personalizados
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #eef0f2; }
    .sidebar .sidebar-content { background-image: linear-gradient(#f8f9fa,#e9ecef); }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES CORE DEL NOTEBOOK (ADAPTADAS) ---

def limpiar_datos_granular(df):
    """Limpieza basada en Bloque 2 del notebook"""
    # Columnas críticas de tu CSV
    cols_num = ['qty', 'net_sale', 'utilidad', 'margen']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    
    # Filtro para evitar errores en logaritmos (Bloque 4)
    df = df.dropna(subset=['prod_nbr', 'qty', 'net_sale'])
    return df[(df['qty'] > 0) & (df['net_sale'] > 0)]

def calcular_elasticidad_subset(df_subset):
    """Calcula elasticidad para los SKUs presentes en el filtro actual"""
    resultados = []
    skus = df_subset['prod_nbr'].unique()
    
    for sku in skus:
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
                'Venta_Total': sub['qty'].sum(),
                'Precio_Prom_Historico': sub['net_sale'].mean(),
                'Margen_Prom_Historico': sub['margen'].mean() if 'margen' in sub.columns else 0
            })
        except: continue
    return pd.DataFrame(resultados)

# --- INTERFAZ DINÁMICA ---

st.title("🎯 Dashboard de Pricing OfficeMax")
st.caption("Filtros Granulares | Basado en CreoQueYaQuedo.ipynb")

# 1. CARGA DE DATOS
f = st.sidebar.file_uploader("1. Cargar Venta_OfficeMax_Ticket.csv", type=['csv'])

if f:
    # Leer y limpiar
    df_raw = pd.read_csv(f)
    df_main = limpiar_datos_granular(df_raw)
    
    # 2. SIDEBAR - FILTROS GRANULARES
    st.sidebar.header("Filtros de Análisis")
    
    # Filtro Departamento
    depto_list = sorted(df_main['dept_nm'].unique().tolist())
    depto_sel = st.sidebar.multiselect("Departamento", depto_list)
    
    df_filtered = df_main.copy()
    if depto_sel:
        df_filtered = df_filtered[df_filtered['dept_nm'].isin(depto_sel)]
    
    # Filtro Categoría (Depende del Depto)
    class_list = sorted(df_filtered['class_nm'].unique().tolist())
    class_sel = st.sidebar.multiselect("Categoría (Clase)", class_list)
    
    if class_sel:
        df_filtered = df_filtered[df_filtered['class_nm'].isin(class_sel)]
        
    # Filtro Marca (Opcional)
    marca_list = sorted(df_filtered['marca'].unique().tolist())
    marca_sel = st.sidebar.multiselect("Marca", marca_list)
    if marca_sel:
        df_filtered = df_filtered[df_filtered['marca'].isin(marca_sel)]

    # --- PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs(["📊 Vista General", "📈 Elasticidad Detallada", "💰 Simulador de Impacto"])

    with tab1:
        st.subheader("Resumen del Segmento Seleccionado")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SKUs en Filtro", df_filtered['prod_nbr'].nunique())
        c2.metric("Venta Qty", int(df_filtered['qty'].sum()))
        c3.metric("Ticket Promedio", f"${df_filtered['net_sale'].mean():.2f}")
        c4.metric("Margen Prom.", f"{df_filtered['margen'].mean()*100:.1f}%")
        
        st.dataframe(df_filtered.head(50), use_container_width=True)

    with tab2:
        st.subheader("Análisis de Demanda por Producto")
        
        # Calculamos elasticidad solo para lo filtrado
        df_elast = calcular_elasticidad_subset(df_filtered)
        
        if not df_elast.empty:
            col_chart, col_table = st.columns([2, 1])
            
            with col_table:
                st.write("Top SKUs por Elasticidad")
                st.dataframe(df_elast[['prod_nbr', 'Elasticidad', 'R2']].sort_values('Elasticidad'), hide_index=True)
            
            with col_chart:
                sku_viz = st.selectbox("Analizar Curva de SKU:", df_elast['prod_nbr'].unique())
                fig = px.scatter(df_filtered[df_filtered['prod_nbr'] == sku_viz], 
                                 x='net_sale', y='qty', trendline="ols",
                                 title=f"Elasticidad Log-Log: SKU {sku_viz}",
                                 labels={'net_sale': 'Precio Unitario', 'qty': 'Cantidad'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos en este filtro para calcular elasticidades (se requieren 5+ registros por SKU).")

    with tab3:
        if 'df_elast' in locals() and not df_elast.empty:
            st.subheader("Simulador de Escenarios Financieros")
            
            # Controles de simulación
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                pct_ajuste = st.select_slider("Ajuste de Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
            with col_s2:
                promo = st.selectbox("Escenario Promocional (Notebook Bloque 6):", ["Ninguno", "2x1", "3x2", "2do al 50%"])
            
            # Proyecciones
            df_sim = df_elast.copy()
            df_sim['Nuevo_Precio'] = df_sim['Precio_Prom_Historico'] * (1 + pct_ajuste)
            df_sim['Nueva_Demanda'] = df_sim['Venta_Total'] * (1 + (df_sim['Elasticidad'] * pct_ajuste))
            
            # Aplicar multiplicadores del notebook
            if promo == "2x1": df_sim['Nueva_Demanda'] *= 1.85
            elif promo == "3x2": df_sim['Nueva_Demanda'] *= 1.55
            
            df_sim['Delta_Ingreso'] = (df_sim['Nuevo_Precio'] * df_sim['Nueva_Demanda']) - (df_sim['Precio_Prom_Historico'] * df_sim['Venta_Total'])
            
            st.dataframe(df_sim[['prod_nbr', 'Elasticidad', 'Precio_Prom_Historico', 'Nuevo_Precio', 'Nueva_Demanda', 'Delta_Ingreso']].style.format(precision=2))
            
            # Dictamen automático
            top_sku = df_sim.iloc[0]
            st.info(f"**Dictamen para el SKU más relevante:** El producto {top_sku['prod_nbr']} tiene una elasticidad de {top_sku['Elasticidad']:.2f}. "
                    f"Un ajuste del {pct_ajuste*100}% resultaría en un cambio de ingreso proyectado de ${top_sku['Delta_Ingreso']:.2f}.")
        else:
            st.error("Carga datos y aplica filtros para activar el simulador.")

else:
    st.info("👋 Bienvenida. Por favor, sube el archivo 'Venta_OfficeMax_Ticket.csv' en el panel de la izquierda para comenzar el análisis granular.")
