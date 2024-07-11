import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
import streamlit as st
import numpy as np
from io import StringIO

#define function for ADF test & KPSS test
def adf_test(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

#define KPSS
def kpss_test(timeseries):
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    
    return kpss_output

def nonStationary(df, column_name):
      
    # Differencing the data
    df["diff_1"] = df[column_name].diff(periods=1)
    df["diff_2"] = df[column_name].diff(periods=2)
    df["diff_3"] = df[column_name].diff(periods=3)   
    
    sales = df['diff_1']

    # Manejar datos como numerios
    sales = pd.to_numeric(sales, errors='coerce')

    # Differencing the data
    sales = sales.diff()

    # Maneja valores faltantes si los hay
    sales = sales.dropna()
    
    p = 5  # AutoRegressive (AR) order
    d = 1  # Differencing order
    q = 0  # Moving Average (MA) order

    model = ARIMA(sales, order=(p, d, q))
    model_fit = model.fit()
    
    forecast_steps = 5

    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_values = forecast.tolist()
    
    last_week = max(df['Week'])
    forecast_weeks = range(last_week + 1, last_week + 1 + forecast_steps)
    last_value = df['Cantidad vendida'].iloc[-1]  # último valor conocido de la serie original
    original_forecast_values = []
    current_value = last_value
    
    for diff_value in forecast_values:
        original_value = diff_value + current_value
        original_forecast_values.append(original_value)
        current_value = original_value
    
    return forecast_weeks, original_forecast_values



def stationary(df, column_name):

    sales = df[column_name]

    # Manejar datos como numerios
    sales = pd.to_numeric(sales, errors='coerce')

    # Differencing the data
    #sales = sales.diff()

    # Maneja valores faltantes si los hay
    sales = sales.dropna()
    
    p = 10  # AutoRegressive (AR) order
    d = 1  # Differencing order
    q = 0  # Moving Average (MA) order

    model = ARIMA(sales, order=(p, d, q))
    model_fit = model.fit()
    
    forecast_steps = 5
    original_forecast_values = []

    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_values = forecast.tolist()
    
    last_week = max(df['Week'])
    forecast_weeks = range(last_week + 1, last_week + 1 + forecast_steps)
    
    
    return forecast_weeks, forecast


def executeForecast(dataframeProducts, dataframeAllData, option):
    
    # Carga de datos con conversión de tipos
    products = dataframeProducts
    products.loc[:,'SKU'] = pd.to_numeric(products['SKU'], errors='coerce')

    # Filtrar y seleccionar en un solo paso
    filtered_products = products.loc[
        products['Categoría'].str.contains(option, case=False, na=False), 
        'SKU'
    ].drop_duplicates()



    data = dataframeAllData
    data.loc[:,'Sku'] = pd.to_numeric(data['Sku'], errors='coerce')
    #data['Sku'] = data['Sku'].astype(str)
    data.loc[:,'Cantidad vendida'] = pd.to_numeric(data['Cantidad vendida'], errors='coerce')
    
    grouper = pd.Grouper(key='Fecha venta', freq='W-MON')
    # Agregar columna con el inicio de la semana  
    data['Week Start'] = data['Fecha venta'].dt.to_period('W-MON').dt.start_time
    
    last_week_start = data['Week Start'].max()
    next_week_starts = [last_week_start + pd.Timedelta(weeks=i) for i in range(1, 6)]

        
    results = []

    for sku in filtered_products:
        print(f'Processing SKU {sku}...')
            
        forecast_values = []
        data_to_plot = 0
        
        data_to_plot = data.query(f"Sku == {sku}")

        # Agregar columna con el número de semana
        data_to_plot['Week'] = data_to_plot.groupby(grouper).ngroup() + 1
        
    

        # Convertir 'Cantidad vendida' a numérico
        data_to_plot['Cantidad vendida'] = pd.to_numeric(data_to_plot['Cantidad vendida'], errors='coerce')

        df = data_to_plot.groupby('Week', as_index=False)['Cantidad vendida'].sum()
        
        if df['Cantidad vendida'].max() == df['Cantidad vendida'].min():
            continue
        
        if len(df) < 5:
            continue


        try:
            # Apply ADF test on the series
            test_one = adf_test(df['Cantidad vendida'].dropna())

            # Apply KPSS test on the series
            test_two = kpss_test(df['Cantidad vendida'].dropna())
        
        except Exception as E:
            continue

        # Compare p-values and make decision
        alpha = 0.05

        if test_one['p-value'] <= alpha and test_two['p-value'] <= alpha:
            print('The series is stationary')
            # Obtener la serie a modelar
            
            try:
                forecast_weeks, original_forecast_values = stationary(df, 'Cantidad vendida')
            except Exception as E:
                print(f"Error {E}")
                continue
            
            
        elif test_one['p-value'] > alpha and test_two['p-value'] > alpha:
            print('The series is non-stationary')
            
            try:
                forecast_weeks, original_forecast_values = nonStationary(df, 'Cantidad vendida')
            except Exception as E:
                print(f"Error {E}")
                continue
            
        elif test_one['p-value'] <= alpha and test_two['p-value'] > alpha:
            print('The series is difference stationary')
            
            try:
                forecast_weeks, original_forecast_values = nonStationary(df, 'Cantidad vendida')
            except Exception as E:
                print(f"Error {E}")
                continue
            
        elif test_one['p-value'] > alpha and test_two['p-value'] <= alpha:
            print('The series is trend stationary')
            
            try:
                forecast_weeks, original_forecast_values = nonStationary(df, 'Cantidad vendida')
            except Exception as E:
                print(f"Error {E}")
                continue
            
        numeric_forecast_values = pd.to_numeric(original_forecast_values, errors='coerce')
        numeric_forecast_values = pd.Series(numeric_forecast_values).replace([np.inf, -np.inf], np.nan)
        
        results.append([sku] +  [str(x) for x in numeric_forecast_values])

    columns = ['SKU'] + next_week_starts
    forecast_df = pd.DataFrame(results, columns=columns)
    
    st.write(forecast_df)

    #forecast_df.to_csv('forecast_results_separated.csv', index=False)

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

st.title('Pronosticos de Ventas (ARIMA)')

option = st.selectbox(
    "¿Qué categoría deseas pronosticar?",
    ("DISPLAY COMPLETO", "TAPAS Y MARCOS", "CENTRO CARGA", "CAMARAS", "BATERIAS"))

st.write("Seleccionaste: ", option)

st.subheader('Calculo de pronosticos :red[*]')
st.markdown("Es necesario que cargue un archivo CSV con los productos a los que deseas hacer el pronostico de ventas, el archivo debe contener las columnas **SKU** y **Categoría**.")
uploaded_file = st.file_uploader("Escoge el archivo de productos", key='products')
uploaded_fileAllData = st.file_uploader("Escoge el archivo de historial de ventas", key='allData')

dataframe = 0
dataframeAllData = 0   

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file, delimiter=',', encoding='utf-8', dtype={'Categoría': str})
    st.write(dataframe.head())
    st.markdown("Antes de continuar verifica que las columnas **SKU** y **Categoría** existan.")
    
if uploaded_fileAllData is not None:
    dataframeAllData = pd.read_csv(uploaded_fileAllData, parse_dates=['Fecha venta'], dayfirst=True)
    st.write(dataframeAllData.head())
    st.markdown("Antes de continuar verifica que las columnas **Sku**, **Cantidad vendida** y **Fecha venta** existan.")
    
if uploaded_file is not None and uploaded_fileAllData is not None:
    if st.button("Run Forecast"):
        executeForecast(dataframe, dataframeAllData, str(option))
    
st.subheader('Actualizacion de ventas')
st.markdown("Es necesario que cargue un archivo CSV con las ventas de los productos, el archivo debe contener las columnas **Sku**, **Fecha venta** y **Cantidad vendida**.")

dataframeNewSales = 0
dataframeHistorySales = 0

historySales = st.file_uploader("Escoge el archivo de historial de ventas", key='historySales')

if historySales is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframeHistorySales = pd.read_csv(historySales, parse_dates=['Fecha venta'], dayfirst=True)
    st.markdown(f" La ultima fecha actualizada es la siguiente YY/MM/DD : :red[{dataframeHistorySales['Fecha venta'].max()}]")
    st.write(dataframeHistorySales.head())
    st.markdown("Antes de continuar verifica que las columnas **Sku**, **Cantidad vendida** y **Fecha venta** existan.")

newsales = st.file_uploader("Escoge el archivo de ventas", key='sales')

if newsales is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframeNewSales = pd.read_csv(newsales, parse_dates=['Fecha venta'], dayfirst=True)
    st.write(dataframeNewSales.head())
    st.markdown("Antes de continuar verifica que las columnas **Sku**, **Cantidad vendida** y **Fecha venta** existan.")


if newsales is not None and historySales is not None:
    if st.button("Update Sales Data"):
        try:
            st.write(merge_files(dataframeNewSales, dataframeHistorySales))
            st.write("Data updated successfully")
        except Exception as e:
            st.write(f"Error: {e}")
