import pandas as pd

# Cargar el nuevo archivo CSV
new_file_path = './outup/clients_per_day.csv'  # Asegúrate de actualizar esta ruta
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

# Guardar el dataframe filtrado en un nuevo archivo CSV
filtered_new_file_path = './outup/clients_per_day_filtro.csv'  # Actualiza esta ruta según sea necesario
data_new_filtered_final.to_csv(filtered_new_file_path, index=False)

print(f"Archivo guardado: {filtered_new_file_path}")
