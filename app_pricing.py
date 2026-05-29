import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# Configuración de página
st.set_page_config(page_title="Pricing Pro Dashboard", layout="wide")

# --- ESTILOS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #dce1e6; }
    .status-green { color: #155724; background-color: #d4edda; padding: 8px; border-radius: 6px; font-weight: bold; }
    .status-yellow { color: #856404; background-color: #fff3cd; padding: 8px; border-radius: 6px; font-weight: bold; }
    .status-red { color: #721c24; background-color: #f8d7da; padding: 8px; border-radius: 6px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DE NEGOCIO ---
def calcular_elasticidad_pro(df):
    resultados = []
    # Columnas mínimas para que el notebook 'creoqueyaquedo' funcione
    req_cols = ['SKU', 'CANTIDAD', 'PRECIO_UNITARIO']
    if not all(col in df.columns for col in req_cols):
        return pd.DataFrame()

    for sku in df['SKU'].unique():
        sub = df[df['SKU'] == sku].copy()
        # Filtramos ceros para evitar errores en logaritmos
        sub = sub[(sub['CANTIDAD'] > 0) & (sub['PRECIO_UNITARIO'] > 0)]
        
        if len(sub) < 5: continue
        
        try:
            sub['log_q'] = np.log(sub['CANTIDAD'])
            sub['log_p'] = np.log(sub['PRECIO_UNITARIO'])
            X = sm.add_constant(sub['log_p'])
            model = sm.OLS(sub['log_q'], X).fit()
            
            resultados.append({
                'SKU': sku,
                'Elasticidad': model.params[1],
                'R2': model.rsquared,
                'P_Value': model.pvalues[1],
                'Precio_Promedio': sub['PRECIO_UNITARIO'].mean()
            })
        except: continue
    return pd.DataFrame(resultados)

# --- ESTRUCTURA DE PESTAÑAS ---
tab1, tab2, tab3 = st.tabs(["📂 Carga & Calidad", "📈 Elasticidad & Análisis", "💰 Pricing & Dictamen"])

with tab1:
    st.header("Carga de Archivos")
    f_ventas = st.file_uploader("Sube tu archivo de Ventas (CSV)", type=['csv'])
    
    if f_ventas:
        df_raw = pd.read_csv(f_ventas)
        total_inicial = len(df_raw)
        
        # Limpieza automática
        df = df_raw.dropna(subset=['SKU', 'CANTIDAD', 'PRECIO_UNITARIO']).copy()
        total_final = len(df)
        eliminados = total_inicial - total_final
        pct_removido = (eliminados / total_inicial) * 100
        
        st.subheader("Auditoría de Datos")
        # Lógica de Semáforo solicitada
        if total_final < 50 or df['SKU'].nunique() < 5 or pct_removido > 50:
            st.markdown("<div class='status-red'>🔴 ROJO: Base insuficiente o muy dañada</div>", unsafe_allow_html=True)
        elif 25 <= pct_removido <= 50:
            st.markdown("<div class='status-yellow'>🟡 AMARILLO: Varianza alta / Pérdida moderada</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-green'>🟢 VERDE: Base óptima para análisis</div>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros Iniciales", total_inicial)
        c2.metric("Registros Limpios", total_final)
        c3.metric("Removidos %", f"{pct_removido:.1f}%")
        
        st.session_state['df_master'] = df

with tab2:
    if 'df_master' in st.session_state:
        df = st.session_state['df_master']
        
        # Sidebar filtros
        st.sidebar.header("Configuración de Vista")
        skus_sel = st.sidebar.multiselect("Selecciona SKUs para comparar", df['SKU'].unique())
        
        # Filtrado de datos para la gráfica
        df_plot = df[df['SKU'].isin(skus_sel)] if skus_sel else df
        
        st.subheader("Visualización de Demanda")
        try:
            # Intentamos con línea de tendencia, si falla, graficamos normal
            fig = px.scatter(df_plot, x='PRECIO_UNITARIO', y='CANTIDAD', color='SKU', 
                             title="Relación Precio vs Cantidad", trendline="ols")
        except:
            fig = px.scatter(df_plot, x='PRECIO_UNITARIO', y='CANTIDAD', color='SKU', 
                             title="Relación Precio vs Cantidad (Sin tendencia por datos insuficientes)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cálculo de elasticidad
        df_elast = calcular_elasticidad_pro(df)
        st.subheader("Tabla de Resultados de Elasticidad")
        st.dataframe(df_elast)
        st.download_button("Descargar CSV Elasticidad", df_elast.to_csv(index=False), "elasticidad.csv")

with tab3:
    if 'df_master' in st.session_state and 'df_elast' in locals():
        st.header("Simulador de Pricing")
        
        col1, col2 = st.columns(2)
        with col1:
            ajuste = st.select_slider("Ajuste de Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
        with col2:
            promo = st.selectbox("Mecánica Especial", ["Ninguna", "2x1", "3x2", "50% en 2da unidad"])

        if not df_elast.empty and skus_sel:
            for s in skus_sel:
                row = df_elast[df_elast['SKU'] == s]
                if not row.empty:
                    e = row['Elasticidad'].values[0]
                    rec = "BAJAR PRECIO / PROMOVER" if e < -1.2 else "SUBIR PRECIO / OPTIMIZAR MARGEN"
                    st.success(f"**Dictamen Analítico SKU {s}:** Elasticidad {e:.2f}. {rec}")
        
        st.info("Utilice los botones para exportar los experimentos de pricing realizados.")
        st.download_button("Exportar Proyecciones", df.to_csv(), "proyecciones_completas.csv")

else:
    st.warning("⚠️ Esperando carga de archivo CSV en la Pestaña 1...")
