import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")

# --- LIMPIEZA DE DATOS PROFUNDA ---
def limpiar_columna_numerica(df, col):
    if col in df.columns:
        # 1. Convertir a string y limpiar caracteres raros
        df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        # 2. Convertir a numérico, lo que no sea número se vuelve NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def calcular_elasticidad_blindada(df):
    resultados = []
    # Solo procesamos si tenemos lo mínimo
    if not all(col in df.columns for col in ['SKU', 'CANTIDAD', 'PRECIO_UNITARIO']):
        return pd.DataFrame()

    for sku in df['SKU'].unique():
        # Filtrar solo el SKU actual y asegurar que NO haya ceros ni vacíos
        sub = df[(df['SKU'] == sku)].copy()
        sub = sub[(sub['CANTIDAD'] > 0) & (sub['PRECIO_UNITARIO'] > 0)].dropna()
        
        if len(sub) < 5:
            continue
        
        try:
            # Forzamos conversión a float64 nativo de numpy para evitar el error de la ufunc
            y = np.log(sub['CANTIDAD'].values.astype(float))
            x = np.log(sub['PRECIO_UNITARIO'].values.astype(float))
            
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            
            resultados.append({
                'SKU': sku, 
                'Elasticidad': model.params[1], 
                'R2': model.rsquared,
                'P_Value': model.pvalues[1]
            })
        except:
            continue
    return pd.DataFrame(resultados)

# --- INTERFAZ ---
tab1, tab2, tab3 = st.tabs(["📁 Datos", "📈 Elasticidad", "💰 Pricing"])

with tab1:
    st.header("Carga de Ventas")
    f = st.file_uploader("Subir CSV", type=['csv'])
    if f:
        df_raw = pd.read_csv(f)
        
        # PROCESO DE LIMPIEZA
        df = limpiar_columna_numerica(df_raw, 'CANTIDAD')
        df = limpiar_columna_numerica(df, 'PRECIO_UNITARIO')
        df = limpiar_columna_numerica(df, 'VENTA_NETA')
        
        # Eliminar cualquier fila que se haya quedado con NaN en columnas críticas
        df = df.dropna(subset=['SKU', 'CANTIDAD', 'PRECIO_UNITARIO'])
        
        st.session_state['df_final'] = df
        st.success(f"✅ Base lista: {len(df)} registros procesados correctamente.")
        st.write("Muestra de datos limpios:", df[['SKU', 'CANTIDAD', 'PRECIO_UNITARIO']].head())

with tab2:
    if 'df_final' in st.session_state:
        df = st.session_state['df_final']
        skus_sel = st.multiselect("Seleccionar SKUs", df['SKU'].unique())
        df_plot = df[df['SKU'].isin(skus_sel)] if skus_sel else df
        
        st.subheader("Análisis de Elasticidad")
        
        # Intentamos graficar; si los datos de tendencia fallan, graficamos solo puntos
        try:
            fig = px.scatter(df_plot, x='PRECIO_UNITARIO', y='CANTIDAD', color='SKU', trendline="ols")
        except:
            fig = px.scatter(df_plot, x='PRECIO_UNITARIO', y='CANTIDAD', color='SKU')
            st.info("Nota: Los datos no permiten trazar línea de tendencia para esta selección.")
            
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de resultados
        df_res = calcular_elasticidad_blindada(df)
        if not df_res.empty:
            st.dataframe(df_res.style.format({'Elasticidad': '{:.2f}', 'R2': '{:.2f}', 'P_Value': '{:.4f}'}))
        else:
            st.warning("No hay suficientes datos válidos para calcular la elasticidad.")

with tab3:
    if 'df_final' in st.session_state:
        st.header("Simulador de Pricing")
        st.info("Mueve el selector para proyectar cambios basados en la elasticidad.")
        ajuste = st.select_slider("Cambio Precio (%)", options=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
        st.download_button("Descargar Reporte Final", df.to_csv(index=False), "pricing_final.csv")
