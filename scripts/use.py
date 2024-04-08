import pandas as pd

# Cargar el archivo CSV
file_path = './outup/usage_over_time.csv'  # Asegúrate de actualizar esta ruta al archivo CSV original
data = pd.read_csv(file_path)

# Convertir la columna 'Time' a datetime
data['Time'] = pd.to_datetime(data['Time'])

# Identificar los días domingos
data['Weekday'] = data['Time'].dt.weekday
data_filtered = data[data['Weekday'] != 6]  # Los domingos son 6

# Lista de festivos en Colombia para 2022 y 2023, formato 'YYYY-MM-DD'
holidays = [
    "2022-01-01", "2022-03-21", "2022-04-14", "2022-04-15", "2022-05-01", "2022-06-20",
    "2022-06-27", "2022-07-04", "2022-07-20", "2022-08-07", "2022-08-15", "2022-10-17",
    "2022-11-07", "2022-11-14", "2022-12-08", "2022-12-25",
    "2023-01-01", "2023-03-20", "2023-04-06", "2023-04-07", "2023-05-01", "2023-06-19",
    "2023-06-26", "2023-07-20", "2023-08-07", "2023-08-21", "2023-10-16", "2023-11-06",
    "2023-11-13", "2023-12-08", "2023-12-25", "2024-01-01", "2024-03-19", "2024-03-28",
    "2024-03-29",
]

# Convertir la lista de festivos a datetime para poder filtrar
holidays_datetime = pd.to_datetime(holidays)

# Filtrar los días festivos
data_filtered_final = data_filtered[~data_filtered['Time'].dt.date.isin(holidays_datetime.date)]

# Eliminar la columna auxiliar 'Weekday'
data_filtered_final.drop('Weekday', axis=1, inplace=True)

# Guardar el dataframe filtrado en un nuevo archivo CSV
filtered_file_path = './outup/usage_over_time_filtrado.csv'  # Actualiza esta ruta según sea necesario
data_filtered_final.to_csv(filtered_file_path, index=False)

print(f"Archivo guardado: {filtered_file_path}")
