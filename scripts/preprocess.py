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

# Ordenar por fecha si aún no lo están
usage_over_time_df.sort_values('Time', inplace=True)
clients_per_day_df.sort_values('Time', inplace=True)

# Resetear índices después de ordenar
usage_over_time_df.reset_index(drop=True, inplace=True)
clients_per_day_df.reset_index(drop=True, inplace=True)

# Mostrar los primeros datos para verificar
print(usage_over_time_df.head())
print(clients_per_day_df.head())

# Guardar los DataFrames limpios en archivos CSV
usage_over_time_df.to_csv('./outup/usage_over_time.csv', index=False)
clients_per_day_df.to_csv('./outup/clients_per_day.csv', index=False)

# A partir de aquí, puedes continuar con el análisis exploratorio,
# la ingeniería de características y la preparación de los datos para el modelo LSTM.

# Cargar el nuevo archivo CSV------ PROCESO QUITAR LOS FESTIVOS Y DOMINGOS
new_file_path = './outup/clients_per_day.csv'  
data_new = pd.read_csv(new_file_path)

# Convertir la columna 'Time' a datetime
data_new['Time'] = pd.to_datetime(data_new['Time'])

# Identificar los días domingos
data_new['Weekday'] = data_new['Time'].dt.weekday
data_new_filtered = data_new[data_new['Weekday'] != 6]  # Los domingos son 6

# Lista de festivos en Colombia para 2022 y 2023, ya definida previamente
holidays = [
    "2022-01-01", "2022-03-21", "2022-04-14", "2022-04-15", "2022-05-01", "2022-06-20",
    "2022-06-27", "2022-07-04", "2022-07-20", "2022-08-07", "2022-08-15", "2022-10-17",
    "2022-11-07", "2022-11-14", "2022-12-08", "2022-12-25",
    "2023-01-01", "2023-03-20", "2023-04-06", "2023-04-07", "2023-05-01", "2023-06-19",
    "2023-06-26", "2023-07-20", "2023-08-07", "2023-08-21", "2023-10-16", "2023-11-06",
    "2023-11-13", "2023-12-08", "2023-12-25", "2024-01-01", "2024-03-19", "2024-03-28",
    "2024-03-29",
]
holidays_datetime = pd.to_datetime(holidays)

# Filtrar los días festivos
data_new_filtered_final = data_new_filtered[~data_new_filtered['Time'].dt.date.isin(holidays_datetime.date)]

# Eliminar la columna auxiliar 'Weekday'
data_new_filtered_final.drop('Weekday', axis=1, inplace=True)

# Guardar el dataframe filtrado en un nuevo archivo CSV ---------
filtered_new_file_path = './outup/clients_per_day_filtro.csv'  
data_new_filtered_final.to_csv(filtered_new_file_path, index=False)

print(f"Archivo guardado: {filtered_new_file_path}")

# Cargar el archivo CSV --------USAGE 
file_path = './outup/usage_over_time.csv'  # Asegúrate de actualizar esta ruta al archivo CSV original
data = pd.read_csv(file_path)

# Convertir la columna 'Time' a datetime
data['Time'] = pd.to_datetime(data['Time'])

# Identificar los días domingos
data['Weekday'] = data['Time'].dt.weekday
data_filtered = data[data['Weekday'] != 6]  # Los domingos son 6

# Filtrar los días festivos
data_filtered_final = data_filtered[~data_filtered['Time'].dt.date.isin(holidays_datetime.date)]

# Eliminar la columna auxiliar 'Weekday'
data_filtered_final.drop('Weekday', axis=1, inplace=True)

# Guardar el dataframe filtrado en un nuevo archivo CSV
filtered_file_path = './outup/usage_over_time_filtrado.csv'  # Actualiza esta ruta según sea necesario
data_filtered_final.to_csv(filtered_file_path, index=False)

print(f"Archivo guardado: {filtered_file_path}")