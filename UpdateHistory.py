import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
import streamlit as st
import numpy as np

def pageUpdate():
    
    st.subheader('Actualizacion de ventas')
    st.markdown("Es necesario que cargue un archivo CSV con las ventas de los productos, el archivo debe contener las columnas **Sku**, **Fecha venta** y **Cantidad vendida**.")

    dataframeNewSales = 0
    dataframeHistorySales = 0

    historySales = st.file_uploader("Escoge el archivo de historial de ventas", key='historySales')

    if historySales is not None:
        # Can be used wherever a "file-like" object is accepted:
        #dataframeHistorySalesTwo = pd.read_csv(historySales, parse_dates=['Fecha venta'], dayfirst=True)
        dataframeHistorySales = pd.read_csv(historySales)
        dataframeHistorySalesTwo = dataframeHistorySales.copy()
        dataframeHistorySalesTwo['Fecha venta'] = pd.to_datetime(dataframeHistorySalesTwo['Fecha venta'], errors='coerce')
        st.write(dataframeHistorySalesTwo['Fecha venta'].head())
        st.markdown(f" La ultima fecha actualizada es la siguiente YY/MM/DD : :red[{dataframeHistorySalesTwo['Fecha venta'].max()}]")
        st.write(dataframeHistorySales.head())
        st.markdown("Antes de continuar verifica que las columnas **Sku**, **Cantidad vendida** y **Fecha venta** existan.")

    newsales = st.file_uploader("Escoge el archivo de ventas", key='sales')

    if newsales is not None:
        # Can be used wherever a "file-like" object is accepted:
        #dataframeNewSales = pd.read_csv(newsales, parse_dates=['Fecha venta'], dayfirst=True)
        dataframeNewSales = pd.read_csv(newsales)
        st.write(dataframeNewSales.head())
        st.markdown("Antes de continuar verifica que las columnas **Sku**, **Cantidad vendida** y **Fecha venta** existan.")


    if newsales is not None and historySales is not None:
        if st.button("Update Sales Data"):
            try:
                st.write(merge_files(dataframeNewSales, dataframeHistorySales))
                st.write("Data updated successfully")
            except Exception as e:
                st.write(f"Error: {e}")
    
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


def merge_files(dataframe, fileTwo):
    # Leer cada archivo CSV y agregarlo a la lista
    df1 = dataframe
    df2 = fileTwo
    
    # Concatenar todos los dataframes en uno solo
    merged_df = pd.concat([df1, df2], ignore_index=True)
    # Eliminar registros duplicados
    merged_df = merged_df.drop_duplicates()
    
    # Guardar el dataframe concatenado en un archivo CSV
    #merged_df.to_csv('All_Data.csv', index=False)
    
    csv = convert_df(merged_df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="./All_Data.csv",
        mime="text/csv",
    )
    
    return merged_df