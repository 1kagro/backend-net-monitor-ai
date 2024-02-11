import pandas as pd
import numpy as np
import os
from glob import glob

# Configura aquí el directorio donde tienes tus archivos Excel
directory_path = './excel_files'

# Encuentra todos los archivos Excel en el directorio
excel_files = glob(os.path.join(directory_path, '*.xlsx'))

# Listas para almacenar los datos de cada archivo
usage_over_time_list = []
clients_per_day_list = []

for file in excel_files:
    # Leer las hojas relevantes
    usage_data = pd.read_excel(file, sheet_name='Usage over time')
    clients_data = pd.read_excel(file, sheet_name='Clients per day')
    
    # Asegúrate de convertir las columnas de tiempo al mismo formato si es necesario
    usage_data['Time'] = pd.to_datetime(usage_data['Time'])
    clients_data['Time'] = pd.to_datetime(clients_data['Time'])
    
    # Agregar los DataFrames a las listas
    usage_over_time_list.append(usage_data)
    clients_per_day_list.append(clients_data)

# Concatenar todos los DataFrames en uno solo
usage_over_time_df = pd.concat(usage_over_time_list, ignore_index=True)
clients_per_day_df = pd.concat(clients_per_day_list, ignore_index=True)

# Limpieza básica de datos
# Aquí puedes añadir cualquier paso específico de limpieza que necesites

# Ordenar por fecha si aún no lo están
usage_over_time_df.sort_values('Time', inplace=True)
clients_per_day_df.sort_values('Time', inplace=True)

# Resetear índices después de ordenar
usage_over_time_df.reset_index(drop=True, inplace=True)
clients_per_day_df.reset_index(drop=True, inplace=True)

# Mostrar los primeros datos para verificar
print(usage_over_time_df.head())
print(clients_per_day_df.head())

# A partir de aquí, puedes continuar con el análisis exploratorio,
# la ingeniería de características y la preparación de los datos para el modelo LSTM.