import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io

# Configuración de la página
st.set_page_config(page_title="Análisis de Outliers", layout="wide")
st.title("📊 Detección de Outliers en el Registro Nacional de Turismo")
st.markdown(""" 
Este análisis se realizó utilizando datos abiertos del portal [Datos.gov.co](https://www.datos.gov.co/resource/sc9w-57a6.json), 
específicamente del conjunto de datos del Registro Nacional de Turismo de la Alcaldía de Valledupar. 
Se aplican estadísticas descriptivas y técnicas de detección de outliers.
""")

# Estilos personalizados con fondo negro para la interfaz y fondo claro para las gráficas
st.markdown("""
    <style>
        body {
            background-color: #111111;  /* Fondo negro */
            color: white;  /* Texto blanco */
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
            padding-right: 3rem;
            padding-left: 3rem;
            padding-bottom: 2rem;
            background-color: #111111;  /* Fondo negro */
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px;
        }
        .stSidebar .sidebar-content {
            background-color: #1c1c1c;  /* Fondo negro para la barra lateral */
            color: white;
        }
        .stRadio>div>label, .stSelectbox>div>label {
            font-size: 14px;
            color: #f0f0f0;  /* Texto blanco */
        }
        .stMarkdown>p {
            font-size: 16px;
            color: #f0f0f0;  /* Texto blanco */
        }
        .stTitle {
            color: #f0f0f0;  /* Texto blanco */
        }
        .stSlider {
            font-size: 14px;
        }
        .stTextInput>div>label {
            color: #f0f0f0;  /* Texto blanco */
        }
    </style>
""", unsafe_allow_html=True)

# Solicitud a la API
url = "https://www.datos.gov.co/resource/sc9w-57a6.json"
params = {"$limit": 483}
response = requests.get(url, params=params)
data = response.json()

# Limpieza y preparación de datos
df = pd.DataFrame(data)
df.columns = df.columns.str.lower().str.strip()

# Eliminar las columnas 'ano', 'cod_mun' y 'cod_dpto' completamente
df = df.drop(columns=['ano', 'cod_mun', 'cod_dpto'])

# Convertir las columnas relevantes a valores numéricos y eliminar nulos
columns_to_convert = ['mes', 'codigo_rnt', 'num_emp1', 'habitaciones', 'camas']
for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])

# Excluir la columna 'ano' de las columnas numéricas (ya eliminada)
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Sidebar con opciones de visualización
st.sidebar.title("Configuración de Visualización")
column_selected = st.sidebar.selectbox("Selecciona una columna para analizar:", numeric_columns)

# Calcular estadísticas descriptivas (total de datos, media, mediana, moda)
statistics = pd.DataFrame(columns=["Total Datos", "Media", "Mediana", "Moda"])

# Datos válidos
total_datos = df[column_selected].dropna().shape[0]
media = df[column_selected].mean()
mediana = df[column_selected].median()
moda = df[column_selected].mode()[0]  # Usamos el primer valor de la moda (en caso de que haya múltiples)

statistics.loc[column_selected] = [total_datos, media, mediana, moda]

# Mostrar estadísticas descriptivas
st.subheader(f"Estadísticas descriptivas para {column_selected}")
st.write(statistics)

# Función para detectar outliers usando el método IQR
def detectar_outliers_iqr(df, columna, umbral=1.5):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - umbral * IQR
    limite_superior = Q3 + umbral * IQR
    outliers_iqr = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    return outliers_iqr

# Función para detectar outliers usando Z-score
def detectar_outliers_z(df, columna):
    z_scores = stats.zscore(df[columna].dropna())
    outliers_z = df[np.abs(z_scores) > 3]  # Valores con Z-score > 3 son outliers
    return outliers_z

# Slider para ajustar el umbral de IQR
umbral_iqr = st.sidebar.slider('Ajusta el umbral de IQR (1.5 por defecto)', 1.0, 5.0, 1.5, 0.1)

# Detección de outliers
outliers_iqr = detectar_outliers_iqr(df, column_selected, umbral_iqr)
outliers_z = detectar_outliers_z(df, column_selected)

# Eliminar outliers de IQR y Z-score
df_sin_outliers_iqr = df[~df.index.isin(outliers_iqr.index)]
df_sin_outliers_z = df[~df.index.isin(outliers_z.index)]

# Función para guardar y devolver una imagen de la gráfica
def save_plot(fig):
    # Guardar la figura en un objeto en memoria
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Función para crear el gráfico de boxplot
def plot_boxplot(data, title, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=data, ax=ax, color=color)
    ax.set_title(title, fontsize=16)
    ax.set_facecolor('white')  # Fondo blanco para el gráfico
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    st.pyplot(fig)  # Mostrar la gráfica en Streamlit
    buf = save_plot(fig)
    return buf

# Función para crear el gráfico de histograma
def plot_histogram(data, title, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data, kde=True, ax=ax, color=color, bins=20)
    ax.set_title(title, fontsize=16)
    ax.set_facecolor('white')  # Fondo blanco para el gráfico
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    st.pyplot(fig)  # Mostrar la gráfica en Streamlit
    buf = save_plot(fig)
    return buf

# Función para crear el gráfico de scatter plot
def plot_scatter(x, y, title, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color=color)
    ax.set_xlabel(x.name, fontsize=12, color='black')
    ax.set_ylabel(y.name, fontsize=12, color='black')
    ax.set_title(title, fontsize=16, color='black')
    ax.set_facecolor('white')  # Fondo blanco para el gráfico
    st.pyplot(fig)  # Mostrar la gráfica en Streamlit
    buf = save_plot(fig)
    return buf

# Interactividad para seleccionar el gráfico
st.sidebar.header("Selecciona el tipo de gráfico")
grafico_selected = st.sidebar.radio("Selecciona el tipo de gráfico:", ("Boxplot", "Histograma", "Scatter Plot"))

# --- Boxplot ---
if grafico_selected == "Boxplot":
    st.subheader(f"Boxplot Original de {column_selected}")
    buf_original = plot_boxplot(df[column_selected], f"Boxplot Original - {column_selected}", "skyblue")
    st.download_button("Descargar Imagen Boxplot Original", buf_original, "boxplot_original.png", "image/png")

    st.subheader(f"Boxplot Limpio (IQR) de {column_selected}")
    buf_iqr = plot_boxplot(df_sin_outliers_iqr[column_selected], f"Boxplot Limpio (IQR) - {column_selected}", "lightgreen")
    st.download_button("Descargar Imagen Boxplot Limpio (IQR)", buf_iqr, "boxplot_iqr.png", "image/png")

    st.subheader(f"Boxplot Limpio (Z-score) de {column_selected}")
    buf_zscore = plot_boxplot(df_sin_outliers_z[column_selected], f"Boxplot Limpio (Z-score) - {column_selected}", "lightcoral")
    st.download_button("Descargar Imagen Boxplot Limpio (Z-score)", buf_zscore, "boxplot_zscore.png", "image/png")

# --- Histograma ---
elif grafico_selected == "Histograma":
    st.subheader(f"Histograma Original de {column_selected}")
    buf_original = plot_histogram(df[column_selected], f"Histograma Original - {column_selected}", "orange")
    st.download_button("Descargar Imagen Histograma Original", buf_original, "histograma_original.png", "image/png")

    st.subheader(f"Histograma Limpio (IQR) de {column_selected}")
    buf_iqr = plot_histogram(df_sin_outliers_iqr[column_selected], f"Histograma Limpio (IQR) - {column_selected}", "lightgreen")
    st.download_button("Descargar Imagen Histograma Limpio (IQR)", buf_iqr, "histograma_iqr.png", "image/png")

    st.subheader(f"Histograma Limpio (Z-score) de {column_selected}")
    buf_zscore = plot_histogram(df_sin_outliers_z[column_selected], f"Histograma Limpio (Z-score) - {column_selected}", "lightcoral")
    st.download_button("Descargar Imagen Histograma Limpio (Z-score)", buf_zscore, "histograma_zscore.png", "image/png")

# --- Scatter Plot ---
elif grafico_selected == "Scatter Plot":
    if len(numeric_columns) > 1:  # Asegurarse de que haya al menos dos columnas para graficar
        other_column = numeric_columns[0] if numeric_columns[0] != column_selected else numeric_columns[1]

        st.subheader(f"Scatter Plot Original: {column_selected} vs {other_column}")
        buf_original = plot_scatter(df[column_selected], df[other_column], f"Scatter Plot Original: {column_selected} vs {other_column}", "green")
        st.download_button("Descargar Imagen Scatter Plot Original", buf_original, "scatter_original.png", "image/png")

        st.subheader(f"Scatter Plot Limpio (IQR): {column_selected} vs {other_column}")
        buf_iqr = plot_scatter(df_sin_outliers_iqr[column_selected], df_sin_outliers_iqr[other_column], f"Scatter Plot Limpio (IQR): {column_selected} vs {other_column}", "lightgreen")
        st.download_button("Descargar Imagen Scatter Plot Limpio (IQR)", buf_iqr, "scatter_iqr.png", "image/png")

        st.subheader(f"Scatter Plot Limpio (Z-score): {column_selected} vs {other_column}")
        buf_zscore = plot_scatter(df_sin_outliers_z[column_selected], df_sin_outliers_z[other_column], f"Scatter Plot Limpio (Z-score): {column_selected} vs {other_column}", "lightcoral")
        st.download_button("Descargar Imagen Scatter Plot Limpio (Z-score)", buf_zscore, "scatter_zscore.png", "image/png")
    else:
        st.write("No hay suficientes columnas numéricas para generar un scatter plot.")
