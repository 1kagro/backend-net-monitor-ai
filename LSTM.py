import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from statsmodels.tsa.seasonal import seasonal_decompose

import holidays

class NET_LSTM:
    
    @classmethod
    def load_and_preprocess_data(cls, file_path: str):
        """
        Load data from a csv file and preprocess it for LSTM model training.
        This includes renaming columns for consistency, converting the 'time'
        column to datetime format, creating new columns for various time components
        (hour, day of week, etc.), and identifying holidays.

        Parameters:
        - file_path: Path to the csv file containing the dataset.

        Returns:
        - A pandas dataframe with the processed data.
        """
        df = pd.read_csv(file_path)
        
        data_cleaned = df.drop_duplicates()  # Drop duplicates
        
        data_cleaned = data_cleaned.drop_duplicates(
            subset='Time')  # Drop duplicates based on 'Time' column
        
        data_cleaned['Time'] = pd.to_datetime(data_cleaned['Time'])  # Convert 'Time' to datetime format
        
        # convert bytes to Gb
        data_cleaned['Total (b/s)'] = data_cleaned['Total (b/s)'] / 1e9
        data_cleaned['Download (b/s)'] = data_cleaned['Download (b/s)'] / 1e9

        # Rename columns
        data_cleaned.rename(columns={
            'Total (b/s)': 'Total (Gb/s)',
            'Download (b/s)': 'Download (Gb/s)'
            }, inplace=True)

        time_intervals_cleaned = data_cleaned['Time'].diff().value_counts() # Check for time intervals
        
        data_cleaned.set_index('Time', inplace=True) # Set 'Time' as index
        
        data_interpolated = data_cleaned.resample('4H').interpolate() # Interpolate missing values, resample to 4-hour intervals

        new_time_intervals = data_interpolated.index.to_series().diff().value_counts() # Check new time intervals
        
        data_interpolated.reset_index(inplace=True) # Reset index
        data_interpolated.head(), new_time_intervals # Display first rows and new time intervals

        col_holidays = holidays.CountryHoliday('CO', years=[2022, 2023, 2024, 2025]) # Colombian holidays
        
        holidays_df = pd.DataFrame(
            sorted(col_holidays.items()), columns=['Date', 'Holiday'])

        holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])

        data_interpolated['Holiday'] = data_interpolated['Time'].dt.date.isin(
            holidays_df['Date'].dt.date).astype(int) # Create 'Holiday' column
        
        data_interpolated.head(), holidays_df.head() # Display first rows of the data and holidays dataframes
        return data_interpolated

    @staticmethod
    def plot_distribution(df: pd.DataFrame, column: str, xlabel: str = 'Total speed (Gb/s)'):
        """
        Plot a histogram of the specified column.
        :param df: a pandas dataframe
        :param column: the column to plot
        :return: None
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(
            df[column],
            bins=30,
            kde=True, 
            color='skyblue',
            edgecolor='black'
        )
        plt.title(f'Distribution of {column}')
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    @staticmethod
    def boxplot(df: pd.DataFrame):
        """
        Plot a boxplot of the specified column.
        :param df: a pandas dataframe
        :param column: the column to plot
        :return: None
        """
        sns.set_theme(style="whitegrid")
        
        fig, axs = plt.subplots(
            nrows=2, 
            figsize=(10, 8),
            constrained_layout=True
            )
        
        sns.boxplot(x=df['Download (Gb/s)'], ax=axs[0])
        axs[0].set_title('Boxplot de Descarga (Gb/s)')
        sns.boxplot(x=df['Total (Gb/s)'], ax=axs[1])
        axs[1].set_title('Boxplot de Total (Gb/s)')

        plt.show()
    
    @staticmethod
    def seasonal_decompose(df: pd.DataFrame):
        """
        Decompose the time series into its components
        :param df: a pandas dataframe
        """
        # Configurar la frecuencia de los datos como diaria (6 registros por día, cada 4 horas)
        
        # Set the 'Time' column as the index of the dataframe
        # data_interpolated.set_index('Time', inplace=True)

        # Decompose the time series into its components, using a period of 6 days (6*24 hours)
        decomposition = seasonal_decompose(df['Total (Gb/s)'], model='additive', period=6*24)
        
        decompose_fig = decomposition.plot()
        decompose_fig.set_size_inches(14, 10)
        plt.show()
    
    @staticmethod
    def scale_data(df: pd.DataFrame):
        """
        Scale the specified columns of the dataframe
        data_interpolated
        """
        features = df[['Download (Gb/s)', 'Total (Gb/s)', 'Holiday']]   # Select the features
        
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
        
        features_scaled_df.head() # Display the first rows of the scaled features
        return features_scaled_df
    
    @staticmethod
    def create_sequences(df: pd.DataFrame, n_steps: int):
        """
        Create sequences
        :param df: a pandas dataframe
        :param n_steps: the number of steps
        :return: a tuple of numpy arrays
        """
        X, y = [], []
        for i in range(len(df) - n_steps):
            sequence = df[i:i + n_steps]
            target = df[i + n_steps]
            X.append(sequence)
            y.append(target)
        return np.array(X), np.array(y)
    
    @staticmethod
    def create_sequences_with_timestamps(df: pd.DataFrame, n_steps: int, target_column: str):
        """
        Generate sequences from the dataframe to be used as inputs for the LSTM model.
        
        Parameters:
        - df: Pandas DataFrame with scaled features.
        - n_steps: Number of steps/records to be used for each input sequence.
        - target_column: Name of the column to predict.

        Returns:
        - Input sequences (X), target values (y), and corresponding timestamps.
        """
        X, y, timestamps = [], [], []
        for i in range(n_steps, len(df)):
            X.append(df.iloc[i-n_steps:i][['download_gb_s', 'total_gb_s', 'hour', 'day_of_week', 'day_of_month', 'month', 'is_holiday']].values)
            y.append(df.iloc[i][target_column])
            timestamps.append(df.iloc[i].name)  # Timestamps for tracking
        return np.array(X), np.array(y), np.array(timestamps)
    
    @staticmethod
    def predict_future_steps(model, last_sequence, n_steps_into_future=42):
        """
        Genera predicciones futuras para un número dado de pasos en el futuro, basado en la última secuencia observada.

        Parámetros:
        - model: El modelo LSTM entrenado.
        - last_sequence: La última secuencia observada, utilizada como punto de partida para las predicciones.
                        Debe tener la forma (n_steps, n_features).
        - n_steps_into_future: Número de pasos en el futuro para predecir. Por defecto 42, cubriendo una semana con intervalos de 4 horas.

        Retorna:
        - Un DataFrame con las predicciones futuras y sus correspondientes marcas de tiempo.
        """
        n_steps_into_future = 42  # Número de predicciones para cubrir una semana.
        future_predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(n_steps_into_future):
            next_step_pred = model.predict(current_sequence[np.newaxis, :, :])
            future_predictions.append(next_step_pred.flatten()[0])
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, :] = next_step_pred

        future_timestamps = pd.date_range(
            start='2024-01-26 00:00:00+00:00', periods=n_steps_into_future, freq='4H')
        predictions_df = pd.DataFrame(future_predictions, index=future_timestamps, columns=['Predicted Network Usage (Gb/s)'])

        return predictions_df

    def create_sequences(input_data, target_column, sequence_length):
        sequences = []
        target = []
        for i in range(len(input_data) - sequence_length):
            # Get the sequence
            seq = input_data[i:i + sequence_length]
            # Get the target for the sequence
            label = input_data.iloc[i + sequence_length][target_column]
            sequences.append(seq)
            target.append(label)
        return np.array(sequences), np.array(target)
    
# Load and preprocess data
file_path = './outup/usage_over_time.csv'
data_interpolated = NET_LSTM.load_and_preprocess_data(file_path)

## Data Exploration

# Plot the distribution of the 'Total (Gb/s)' column and the 'Download (Gb/s)' column
NET_LSTM.boxplot(data_interpolated)

# Plot the distribution of the 'Total (Gb/s)' column
NET_LSTM.seasonal_decompose(data_interpolated)


features_scaled_df = NET_LSTM.scale_data(data_interpolated)

sequence_length = 48 # Number of time steps based on 4-hour intervals (48 * 4h = 192h) = 8 days
features_sequences, target_sequences = NET_LSTM.create_sequences(features_scaled_df, 'Total (Gb/s)', sequence_length)

# Split the data into training, validation, and test sets, using an 80-10-10 split
split_train = int(0.8 * len(features_sequences))
split_val = int(0.9 * len(features_sequences))

X_train, Y_train = features_sequences[:split_train], target_sequences[:split_train]
X_val, Y_val = features_sequences[split_train:split_val], target_sequences[split_train:split_val]
X_test, Y_test = features_sequences[split_val:], target_sequences[split_val:]

# Display the shapes of the training, validation, and test sets
X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape

# Build the model
model = Sequential([
    LSTM(50, input_shape=(
        X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error') # Compile the model

model.summary() # Display the model summary

# Train the model
history = model.fit(
    X_train, Y_train, epochs=50, batch_size=32,
    validation_data=(X_val, Y_val), verbose=1)

# Evaluate the model
test_loss = model.evaluate(X_test, Y_test)

model.save('new_model_lstm.keras')

# Build the model
# model = Sequential([
#     Input(shape=(n_steps, n_features)),
#     LSTM(128, activation='relu', return_sequences=True),
#     Dropout(0.2),
#     LSTM(64, activation='relu'),
#     Dropout(0.2),
#     Dense(1)
# ])

# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.summary()

## Define callbacks

# # Early stopping
# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     mode='min',
#     restore_best_weights=True
# )

# # Model checkpoint
# model_checkpoint = ModelCheckpoint(
#     'best_model11.keras',
#     monitor='val_loss',
#     save_best_only=True,
#     # save_weights_only=False,
#     # mode='min'
# )

# # Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=100,
#     batch_size=64,
#     callbacks=[early_stopping, model_checkpoint],
#     verbose=1
# )

# test_loss = model.evaluate(X_test, y_test)
# print(f'Test Loss: {test_loss}')

# predictions = model.predict(X_test)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show()

# # Calculate and pritn error metrics
# mae = mean_absolute_error(y_test, predictions)
# mse = mean_squared_error(y_test, predictions)
# rmse = np.sqrt(mse)  # Correct RMSE calculation

# print("Mean Absolute Error", mae)
# print("Mean Squared Error", mse)
# print("Root Mean Squared Error", rmse)

# # Visualization of the Predictions vs Real Values
# plt.figure(figsize=(10, 6))
# plt.plot(timestamps_test[-42:], y_test[-42:], label='Real', marker='.')
# plt.plot(timestamps_test[-42:], predictions[-42:], label='Predicted', linestyle='--', marker='.')
# plt.title('Usage Over Time: Real vs Predicted')
# plt.xlabel('Time')
# plt.ylabel('Network Usage (Gb/s)')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45) # Rotate x-axis labels for better readability
# plt.tight_layout() # Ensure the labels fit within the figure area
# plt.show()

# n_steps_into_future = 42  # Number of steps into the future
# future_predictions = NET_LSTM.predict_future_steps(model, X_test[-1], n_steps_into_future)

# future_timestamps = pd.date_range(start=timestamps_test[-1], periods=n_steps_into_future + 1, freq='4h')[1:]

# plt.figure(figsize=(10, 6))
# plt.plot(future_timestamps[-42:], future_predictions[-42:], label='Future Predictions', marker='o', linestyle='--', color='red')
# plt.title('Future Network Usage Predictions')
# plt.xlabel('Time')
# plt.ylabel('Network Usage (Gb/s)')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()