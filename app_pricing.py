import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Pricing Granular OfficeMax", layout="wide")

# --- FUNCIONES CORE ---

def limpiar_datos_officemax(df):
    """Limpieza estricta basada en el Bloque 2 del notebook"""
    cols_num = ['qty', 'net_sale', 'margen', 'utilidad']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    
    # Filtro para evitar errores en logaritmos
    df = df.dropna(subset=['prod_nbr', 'qty', 'net_sale'])
    return df[(df['qty'] > 0) & (df['net_sale'] > 0)]

def calcular_elasticidad_granular(df_filtered):
    """Cálculo de elasticidad OLS (Bloque 4 del notebook)"""
    resultados = []
    # Calculamos para los productos que sobrevivieron a los filtros
    for sku in df_filtered['prod_nbr'].unique():
        sub = df_filtered[df_filtered['prod_nbr'] == sku]
        if len(sub) < 5: continue # Mínimo estadístico
        
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
                'Precio_Prom_Real': sub['net_sale'].mean(),
                'Margen_Prom': sub['margen'].mean() if 'margen' in sub.columns else 0
            })
        except: continue
    return pd.DataFrame(resultados)

# --- INTERFAZ DINÁMICA ---

st.sidebar.title("📊 Filtros OfficeMax")
f = st.sidebar.file_uploader("Cargar Venta_OfficeMax_Ticket.csv", type=['csv'])

if f:
    # 1. Carga Inicial
    df_raw = pd.read_csv(f)
    df_main = limpiar_datos_officemax(df_raw)
    
    # 2. Lógica de Filtros Granulares (con Keys únicas para evitar DuplicateWidgetID)
    st.sidebar.markdown("---")
    
    # Filtro Departamento
    deptos = sorted(df_main['dept_nm'].unique())
    sel_depto = st.sidebar.multiselect("Departamento", deptos, key="m_depto")
    
    temp_df = df_main.copy()
    if sel_depto:
        temp_df = temp_df[temp_df['dept_nm'].isin(sel_depto)]
        
    # Filtro Clase (dependiente del Depto)
    clases = sorted(temp_df['class_nm'].unique())
    sel_clase = st.sidebar.multiselect("Categoría / Clase", clases, key="m_clase")
    if sel_clase:
        temp_df = temp_df[temp_df['class_nm'].isin(sel_clase)]
        
    # Filtro Marca (dependiente de Clase)
    marcas = sorted(temp_df['marca'].unique())
    sel_marca = st.sidebar.multiselect("Marca", marcas, key="m_marca")
    if sel_marca:
        temp_df = temp_df[temp_df['marca'].isin(sel_marca)]

    # 3. Tabs Principales
    tab1, tab2, tab3 = st.tabs(["📋 Resumen", "📈 Análisis de Demanda", "💰 Simulador"])

    with tab1:
        st.subheader("Métricas del Segmento Seleccionado")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SKUs Filtrados", temp_df['prod_nbr'].nunique())
        c2.metric("Volumen Total", f"{int(temp_df['qty'].sum()):,}")
        c3.metric("Precio Prom.", f"${temp_df['net_sale'].mean():.2f}")
        c4.metric("Margen Prom.", f"{temp_df['margen'].mean()*100:.1f}%")
        
        st.dataframe(temp_df.head(100), use_container_width=True)

    with tab2:
        st.subheader("Cálculo de Elasticidad por Producto")
        df_elast = calcular_elasticidad_granular(temp_df)
        
        if not df_elast.empty:
            col_graph, col_data = st.columns([2, 1])
            
            with col_data:
                st.write("Top Productos")
                st.dataframe(df_elast[['prod_nbr', 'Elasticidad', 'R2']].sort_values('Elasticidad'), hide_index=True)
            
            with col_graph:
                sku_viz = st.selectbox("Seleccionar SKU para visualizar:", df_elast['prod_nbr'].unique(), key="s_viz")
                fig = px.scatter(temp_df[temp_df['prod_nbr'] == sku_viz], 
                                 x='net_sale', y='qty', trendline="ols",
                                 title=f"Elasticidad: SKU {sku_viz}",
                                 labels={'net_sale': 'Precio ($)', 'qty': 'Cantidad (U)'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Selecciona un filtro con más datos. Se requieren al menos 5 puntos de venta por SKU para calcular la elasticidad.")

    with tab3:
        if 'df_elast' in locals() and not df_elast.empty:
            st.subheader("Simulador de Impacto (Notebook Bloque 6)")
            
            # Ajustes globales para el segmento
            ajuste_p = st.select_slider("Ajuste de Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15], value=0.0)
            promo = st.selectbox("Mecánica Promocional:", ["Ninguna", "2x1", "3x2", "2do al 50%"], key="s_promo")
            
            df_sim = df_elast.copy()
            df_sim['Nuevo_Precio'] = df_sim['Precio_Prom_Real'] * (1 + ajuste_p)
            df_sim['Nueva_Demanda'] = df_sim['Venta_Acumulada'] * (1 + (df_sim['Elasticidad'] * ajuste_p))
            
            # Multiplicadores del notebook
            if promo == "2x1": df_sim['Nueva_Demanda'] *= 1.85
            elif promo == "3x2": df_sim['Nueva_Demanda'] *= 1.55
            elif promo == "2do al 50%": df_sim['Nueva_Demanda'] *= 1.30
            
            df_sim['Delta_Venta_$'] = (df_sim['Nuevo_Precio'] * df_sim['Nueva_Demanda']) - (df_sim['Precio_Prom_Real'] * df_sim['Venta_Acumulada'])
            
            st.dataframe(df_sim[['prod_nbr', 'Elasticidad', 'Precio_Prom_Real', 'Nuevo_Precio', 'Nueva_Demanda', 'Delta_Venta_$']].style.format(precision=2))
            
            # Dictamen
            st.markdown("### 💡 Insight Estratégico")
            avg_e = df_sim['Elasticidad'].mean()
            if avg_e < -1.1:
                st.success(f"El segmento seleccionado es **Elástico** ({avg_e:.2f}). Se recomienda bajar precios o aplicar la promoción '{promo}'.")
            else:
                st.info(f"El segmento es **Inelástico** ({avg_e:.2f}). Hay oportunidad de subir precio sin perder tanto volumen.")
        else:
            st.info("Carga datos para activar el simulador financiero.")

else:
    st.info("👋 Bienvenida. Por favor sube el archivo 'Venta_OfficeMax_Ticket.csv' en el panel lateral.")
