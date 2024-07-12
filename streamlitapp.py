import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
import streamlit as st
import numpy as np

from ForecastArima import pageForecast
from UpdateHistory import pageUpdate
from SubsidiaryForecast import subsidiaryForecast

pg = st.navigation([
    st.Page(pageForecast, title="Calculo de Pronosticos", icon=":material/data_thresholding:"),
    st.Page(pageUpdate, title="Actualizacion de Ventas", icon=":material/update:"),
    st.Page(subsidiaryForecast, title="Pronostico P/Sucursal", icon=":material/data_thresholding:")
])

pg.run()