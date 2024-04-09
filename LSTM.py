import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df.sort_values('time', inplace=True)  # Ensure data is ordered by time
        # df.index = df.index.tz_convert('America/Bogota')
        
        df['download_gb_s'] = df['download_b/s'] / (10**9)
        df['total_gb_s'] = df['total_b/s'] / (10**9)
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek  # Extract day of the week (0=Monday, 6=Sunday)
        df['day_of_month'] = df.index.day  # Extract day of the month (1-31)
        df['month'] = df.index.month
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Identify Colombian holidays
        col_holidays = holidays.Colombia()
        df['is_holiday'] = [1 if date in col_holidays else 0 for date in df.index.date]

        return df

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
    def scale_data(df: pd.DataFrame, columns: list, target_column: str = 'total_gb_s'):
        """
        Scale the specified columns of the dataframe
        """
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df[columns])
        df_scaled = pd.DataFrame(df_scaled, columns=columns, index=df.index)
        return df_scaled
    
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

# Load and preprocess data
file_path = './outup/usage_over_time.csv'
df = NET_LSTM.load_and_preprocess_data(file_path)

# Scale the data
columns_to_scale = ['download_gb_s', 'total_gb_s', 'hour',
                    'day_of_week', 'day_of_month', 'month', 'is_weekend', 'is_holiday']
df_scaled = NET_LSTM.scale_data(df, columns_to_scale)  # Scale the data

# Visualize the distribution of the total speed (Gb/s)
NET_LSTM.plot_distribution(df, 'total_gb_s')  # Plot histogram
# df = pd.get_dummies(df, columns=['hour', 'day_of_week', 'day_of_month', 'month'])
# df.head()

plt.figure(figsize=(16, 6))
# Graphical representation of the distribution of the total network usage by hour of the day
plt.subplot(1, 2, 1)
sns.boxplot(x='hour', y='total_gb_s', data=df,)
plt.title('Distribución del Uso Total de la Red por Hora del Día')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Network Usage (Gb/s)')

# Graphical representation of the distribution of the total network usage by day of the week
plt.subplot(1, 2, 2)
sns.boxplot(x='day_of_week', y='total_gb_s', data=df)
plt.title('Distribución del Uso Total de la Red por Día de la Semana')
plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
plt.ylabel('Total Network Usage (Gb/s)')

plt.tight_layout()
plt.show()


required_cols = ['download_gb_s', 'total_gb_s', 'hour', 'day_of_week', 'day_of_month', 'month', 'is_holiday']
df = df[required_cols]
df.head()

df.isna().sum()
df = df.ffill()  # Fill missing values

# scaled_features = NET_LSTM.scale_data(df)  # Scale the data
# Number of time steps based on 7 days (24h/4h = 6 * 7d = 42)
n_steps = 42

# Create sequences (Divide the data into input and target)
X, y, timestamps = NET_LSTM.create_sequences_with_timestamps(df_scaled, n_steps, 'total_gb_s')

train_size = int(len(X) * 0.7)  # 70% train, 20% validation, 10% test
val_size = int(len(X) * 0.2)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]
timestamps_test = timestamps[train_size + val_size:] # Store the timestamps for the test set

# X_train.shape, X_val.shape, X_test.shape
# y_train.shape, y_val.shape, y_test.shape

n_features = X_train.shape[2]  # Number of features

# Build the model
model = Sequential([
    Input(shape=(n_steps, n_features)),
    LSTM(128, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

## Define callbacks

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    restore_best_weights=True
)

# Model checkpoint
model_checkpoint = ModelCheckpoint(
    'best_model11.keras',
    monitor='val_loss',
    save_best_only=True,
    # save_weights_only=False,
    # mode='min'
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

predictions = model.predict(X_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Calculate and pritn error metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)  # Correct RMSE calculation

print("Mean Absolute Error", mae)
print("Mean Squared Error", mse)
print("Root Mean Squared Error", rmse)

# Visualization of the Predictions vs Real Values
plt.figure(figsize=(10, 6))
plt.plot(timestamps_test[-42:], y_test[-42:], label='Real', marker='.')
plt.plot(timestamps_test[-42:], predictions[-42:], label='Predicted', linestyle='--', marker='.')
plt.title('Usage Over Time: Real vs Predicted')
plt.xlabel('Time')
plt.ylabel('Network Usage (Gb/s)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45) # Rotate x-axis labels for better readability
plt.tight_layout() # Ensure the labels fit within the figure area
plt.show()

n_steps_into_future = 42  # Number of steps into the future
future_predictions = NET_LSTM.predict_future_steps(model, X_test[-1], n_steps_into_future)

future_timestamps = pd.date_range(start=timestamps_test[-1], periods=n_steps_into_future + 1, freq='4h')[1:]

plt.figure(figsize=(10, 6))
plt.plot(future_timestamps[-42:], future_predictions[-42:], label='Future Predictions', marker='o', linestyle='--', color='red')
plt.title('Future Network Usage Predictions')
plt.xlabel('Time')
plt.ylabel('Network Usage (Gb/s)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()