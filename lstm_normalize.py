import pandas as pd
from sklearn.preprocessing import MinMaxScaler

usage_over_time_df = pd.read_csv('usage_over_time.csv')
clients_per_day_df = pd.read_csv('clients_per_day.csv')

# Sort by Time if not already sorted and reset indexes
usage_over_time_df.sort_values('Time', inplace=True)
clients_per_day_df.sort_values('Time', inplace=True)
usage_over_time_df.reset_index(drop=True, inplace=True)
clients_per_day_df.reset_index(drop=True, inplace=True)

# Normalization of numerical features using MinMaxScaler
scaler_usage = MinMaxScaler()
scaler_clients = MinMaxScaler()

# Normalize 'Download (b/s)' and 'Total (b/s)' for usage_over_time_df
usage_over_time_df[['Download (b/s)', 'Total (b/s)']] = scaler_usage.fit_transform(
    usage_over_time_df[['Download (b/s)', 'Total (b/s)']])

# Normalize '# Clients' for clients_per_day_df
clients_per_day_df[['# Clients']] = scaler_clients.fit_transform(
    clients_per_day_df[['# Clients']])

