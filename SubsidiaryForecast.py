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
    
    forecast_steps = 4

    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_values = forecast.tolist()
    
    last_week = max(df['Week'])
    forecast_weeks = range(int(last_week) + 1, int(last_week) + 1 + forecast_steps)
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
    
    forecast_steps = 4
    original_forecast_values = []

    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_values = forecast.tolist()
    
    
    last_week = max(df['Week'])
    forecast_weeks = range(int(last_week) + 1, int(last_week) + 1 + forecast_steps)
    
    
    return forecast_weeks, forecast_values


def executeForecast(dataframeProducts, dataframeAllData, categoria, sucursal):
    
        # Carga de datos con conversión de tipos
    products = dataframeProducts.copy()
    products.loc[:,'SKU'] = pd.to_numeric(products['SKU'], errors='coerce')

    # Filtrar y seleccionar en un solo paso
    filtered_products = products.loc[
        products['Categoría'].str.contains(categoria, case=False, na=False), 
        'SKU'
    ].drop_duplicates()

    grouper = pd.Grouper(key='Fecha venta', freq='W-MON')

    data = dataframeAllData.copy()
    data.loc[:,'Sku'] = pd.to_numeric(data['Sku'], errors='coerce')
    data.loc[:,'Cantidad vendida'] = pd.to_numeric(data['Cantidad vendida'], errors='coerce')
    
    
    # Convertir 'Fecha venta' a datetime si no está en ese formato
    data['Fecha venta'] = pd.to_datetime(data['Fecha venta'], errors='coerce')

    # Agregar columna con el inicio de la semana
    data['Week'] = data.groupby(grouper).ngroup() + 1
    data['Week Start'] = data['Fecha venta'].dt.to_period('W-MON').dt.start_time
    
    data2 = data.loc[
            data['Sucursal'].str.contains(sucursal, case=False, na=False)
    ].drop_duplicates()
    
    data = data2
    
    first_week_start = data['Week Start'].min()
    last_week_start = data['Week Start'].max()
    

    
    next_week_starts = [last_week_start + pd.Timedelta(weeks=i) for i in range(1, 5)]

    results = []
    
    grouper = pd.Grouper(key='Fecha venta', freq='W-MON')
    all_weeks = pd.DataFrame({'Fecha venta': pd.date_range(start=first_week_start, end=last_week_start, freq='W-MON')})
    all_weeks['Week'] = all_weeks.groupby(grouper).ngroup() + 1

    for sku in filtered_products:
        if sku < 1000:
            continue

        data_to_plot = data[data['Sku'] == sku].copy()

        # Convertir 'Cantidad vendida' a numérico
        data_to_plot['Cantidad vendida'] = pd.to_numeric(data_to_plot['Cantidad vendida'], errors='coerce')

        default_rows = []
        # Llenar las semanas faltantes con valores por defecto
        for week_start in all_weeks['Fecha venta']:
            if not ((data_to_plot['Week Start'] == week_start).any()):
                default_row = {
                    'Sucursal': sucursal,
                    'Sku': sku,  # Cambia 'default_sku' según tus necesidades
                    'Modelo': '',
                    'Nombre producto': '',
                    'Folio de venta': '',
                    'Cliente': 'PUBLICO EN GENERAL',
                    'Fecha venta': week_start,
                    'Cantidad vendida': 0.0,
                    'Importe s/impuestos': 0.0,
                    'Week': all_weeks.loc[all_weeks['Fecha venta'] == week_start, 'Week'].values[0],
                    'Week Start': week_start
                }
                default_rows.append(default_row)
                
        if default_rows:
            default_df = pd.DataFrame(default_rows)
            data_to_plot = pd.concat([data_to_plot, default_df], ignore_index=True)

        data_to_plot = data_to_plot.sort_values(by='Week Start')
        
        
        data_to_plot = data_to_plot.groupby('Week', as_index=False)['Cantidad vendida'].sum()
        
        df = data_to_plot.copy()
        
        #df.to_csv('example_grouped.csv', index=False)
        
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
            # print('The series is stationary')
            # Obtener la serie a modelar
            
            try:
                
                forecast_weeks, original_forecast_values = stationary(df, 'Cantidad vendida')
            
                original_forecast_values = [round(value) for value in original_forecast_values]
                original_forecast_values.append(sum(original_forecast_values))
                
                numeric_forecast_values = pd.to_numeric(original_forecast_values, errors='coerce')
                numeric_forecast_values = pd.Series(numeric_forecast_values).replace([np.inf, -np.inf], np.nan)
                
                results.append([sku] +  [str(x) for x in numeric_forecast_values])
                
            except:
                continue
            
            
        elif test_one['p-value'] > alpha and test_two['p-value'] > alpha:
            #print('The series is non-stationary')
            
            try:
                
                forecast_weeks, original_forecast_values = nonStationary(df, 'Cantidad vendida')
            
                original_forecast_values = [round(value) for value in original_forecast_values]
                original_forecast_values.append(sum(original_forecast_values))
                    
                numeric_forecast_values = pd.to_numeric(original_forecast_values, errors='coerce')
                numeric_forecast_values = pd.Series(numeric_forecast_values).replace([np.inf, -np.inf], np.nan)
                    
                results.append([sku] +  [str(x) for x in numeric_forecast_values])
                
            except:
                continue
            
        elif test_one['p-value'] <= alpha and test_two['p-value'] > alpha:
            #print('The series is difference stationary')
            
            try:
                
                forecast_weeks, original_forecast_values = nonStationary(df, 'Cantidad vendida')
            
                original_forecast_values = [round(value) for value in original_forecast_values]
                original_forecast_values.append(sum(original_forecast_values))
                    
                numeric_forecast_values = pd.to_numeric(original_forecast_values, errors='coerce')
                numeric_forecast_values = pd.Series(numeric_forecast_values).replace([np.inf, -np.inf], np.nan)
                    
                results.append([sku] +  [str(x) for x in numeric_forecast_values])
                
            except:
                continue
            
        elif test_one['p-value'] > alpha and test_two['p-value'] <= alpha:
            #print('The series is trend stationary')
            
            try:
                
                forecast_weeks, original_forecast_values = nonStationary(df, 'Cantidad vendida')
            
                original_forecast_values = [round(value) for value in original_forecast_values]
                original_forecast_values.append(sum(original_forecast_values))
                
                numeric_forecast_values = pd.to_numeric(original_forecast_values, errors='coerce')
                numeric_forecast_values = pd.Series(numeric_forecast_values).replace([np.inf, -np.inf], np.nan)
                    
                results.append([sku] +  [str(x) for x in numeric_forecast_values])
                
            except:
                continue
                
            
            

    columns = ['SKU'] + next_week_starts + ['Total']
    forecast_df = pd.DataFrame(results, columns=columns)
    
    st.write(forecast_df)


def subsidiaryForecast():
    
    st.subheader('Pronostico P/Sucursal')
    
    categoria = st.selectbox(
        "CATEGORIA",
        ("DISPLAY COMPLETO", "TAPAS Y MARCOS", "CENTRO CARGA", "CAMARAS", "BATERIAS"))

    st.write("Seleccionaste: ", categoria)
    
    sucursal = st.selectbox(
        "SUCURSAL",
        ("MONTERREY", "TIJUANA", "QUERETARO", "PUEBLA", "CEDIS"))

    st.write("Seleccionaste: ", sucursal)
    
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
            executeForecast(dataframe, dataframeAllData, str(categoria), str(sucursal))
    
    