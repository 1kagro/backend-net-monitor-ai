import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class NET_LSTM:
    
    @classmethod
    def load_and_preprocess_data(cls, file_path: str):
        """
        Load data from a csv file
        :param file_path: path to the csv file
        :return: a pandas dataframe
        """
        df = pd.read_csv(file_path)
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        # Convert to datetime
        df['time'] = pd.to_datetime(df['time'])
        # Convert b/s to Gb/s for relevant columns
        df['download_gb_s'] = df['download_b/s'] / (10**9)
        df['total_gb_s'] = df['total_b/s'] / (10**9)
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
    def scale_data(df: pd.DataFrame, columns: list):
        """
        Scale the specified columns of the dataframe
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = df.copy()
        df_scaled[columns] = scaler.fit_transform(df[columns])
        return df_scaled, scaler
    
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
        X, y, timestamps = [], [], []
        for i in range(n_steps, len(df)):
            X.append(df.iloc[i-n_steps:i][['download_gb_s', 'total_gb_s']].values)
            y.append(df.iloc[i][target_column])
            timestamps.append(df.iloc[i]['time'])  # Store the timestamp
        return np.array(X), np.array(y), np.array(timestamps)

# Load and preprocess data
file_path = './outup/usage_over_time.csv'
df = NET_LSTM.load_and_preprocess_data(file_path)

# Scale the data
columns_to_scale = ['download_gb_s', 'total_gb_s']
df_scaled, scaler = NET_LSTM.scale_data(df, columns_to_scale)  # Scale the data

# Visualize the distribution of the total speed (Gb/s)
NET_LSTM.plot_distribution(df, 'total_gb_s')  # Plot histogram


df.index = df['time']

time_series = df.index # Extract time series

# df['hour'] = df.index.hour  # Extract hour
# df['day_of_week'] = df.index.dayofweek  # Extract day of the week
# df['day_of_month'] = df.index.day  # Extract day of the month
# df['month'] = df.index.month  # Extract month

# df = pd.get_dummies(df, columns=['hour', 'day_of_week', 'day_of_month', 'month'])
# df.head()

required_cols = ['download_gb_s', 'total_gb_s']
df = df[required_cols]
df.head()

df.isna().sum()
df = df.ffill()  # Fill missing values

# df_final = df.resample('D').mean() # Resample data, daily
# df_final.head()

# # df.isna().sum() # Check missing values
# df_final = df.ffill()  # Fill missing values
# df_final.head()

# scaled_features = NET_LSTM.scale_data(df)  # Scale the data
n_steps = 18  # Number of time steps based on 3 days
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
    LSTM(100, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
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
    'best_model1.keras',
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
    batch_size=32,
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
rmse = np.sqrt(mae) # Correct RMSE calculation

print("Mean Absolute Error", mae)
print("Mean Squared Error", mse)
print("Root Mean Squared Error", rmse)

# Visualization of the Predictions vs Real Values
plt.figure(figsize=(10, 6))
plt.plot(timestamps_test, y_test, label='Real', color='blue', marker='.')
plt.plot(timestamps_test, predictions.flatten(), label='Predicted', linestyle='--', marker='.')
plt.title('Usage Over Time: Real vs Predicted')
plt.xlabel('Time')
plt.ylabel('Network Usage (Gb/s)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45) # Rotate x-axis labels for better readability
plt.tight_layout() # Ensure the labels fit within the figure area
plt.show()