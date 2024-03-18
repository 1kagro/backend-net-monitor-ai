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
    def load_data(cls, file_path: str):
        """
        Load data from a csv file
        :param file_path: path to the csv file
        :return: a pandas dataframe
        """
        df = pd.read_csv(file_path)
        df = cls._clean_data(df)
        return df

    @staticmethod
    def _clean_data(df: pd.DataFrame):
        """
        Clean the data
        :param df: a pandas dataframe
        :return: a pandas dataframe
        """
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        return df

    @staticmethod
    def plot_hist_d(df: pd.DataFrame, column: str):
        """
        Plot a histogram 
        :param df: a pandas dataframe
        :param column: the column to plot
        :return: None
        """
        if column == 'total_b/s':
            df['download_gb_s'] = df[column] / (10**9)  # Convert to Gb/s
            df['total_gb_s'] = df['total_b/s'] / (10**9)  # Convert to Gb/s
            df['time'] = pd.to_datetime(df['time'])  # Convert to datetime
            
            plt.figure(figsize=(10, 6))
            plt.hist(
                df['total_gb_s'],
                bins=30, color='skyblue',
                edgecolor='black'
            )
            plt.title('Distribución de la Velocidad Total (Descarga y subida)')
            plt.xlabel('Velocidad de total (Gb/s)')
            plt.ylabel('Frecuencia')
            plt.grid(axis='y', alpha=0.75)
        plt.show()

    @staticmethod
    def scale_data(df: pd.DataFrame):
        """
        Scale the data
        :param df: a pandas dataframe
        :return: a pandas dataframe
        """
        # copilot
        # scaler = MinMaxScaler()
        # df_scaled = pd.DataFrame(
        #     scaler.fit_transform(df),
        #     columns=df.columns
        # )
        # return df_scaled
        
        # Select the features to normalize
        features = df['total_gb_s'].values.reshape(-1, 1)
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        return scaled_features
        
    @staticmethod
    def create_sequences(df: pd.DataFrame, n_steps: int):
        """
        Create sequences
        :param df: a pandas dataframe
        :param n_steps: the number of steps
        :return: a tuple of numpy arrays
        """
        # copilot
        # X, y = [], []
        # for i in range(len(df)):
        #     end_ix = i + n_steps
        #     if end_ix > len(df) - 1:
        #         break
        #     seq_x, seq_y = df[i:end_ix], df[end_ix]
        #     X.append(seq_x)
        #     y.append(seq_y)
        # return np.array(X), np.array(y)

        X, y = [], []
        for i in range(len(df) - n_steps):
            sequence = df[i:i + n_steps]
            target = df[i + n_steps]
            X.append(sequence)
            y.append(target)
        return np.array(X), np.array(y)
    
df = NET_LSTM.load_data('./outup/usage_over_time.csv')
NET_LSTM.plot_hist_d(df, 'total_b/s')  # Plot histogram

df.index = df['time']
df['hour'] = df.index.hour  # Extract hour
df['day_of_week'] = df.index.dayofweek  # Extract day of the week
df['day_of_month'] = df.index.day  # Extract day of the month
df['month'] = df.index.month  # Extract month

df = pd.get_dummies(df, columns=['hour', 'day_of_week', 'day_of_month', 'month'])
df.head()

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

scaled_features = NET_LSTM.scale_data(df)  # Scale the data
n_steps = 18  # Number of time steps based on 3 days
X, y = NET_LSTM.create_sequences(scaled_features, n_steps) # Create sequences (Divide the data into input and target)

train_size = int(len(X) * 0.7)  # 70% train, 20% validation, 10% test
val_size = int(len(X) * 0.2)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# X_train.shape, X_val.shape, X_test.shape
# y_train.shape, y_val.shape, y_test.shape

n_features = df.shape[1]  # Number of features

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
    'best_model.keras',
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


mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mae)

print("Mean Absolute Error", mae)
print("Mean Squared Error", mse)
print("Root Mean Squared Error", rmse)

# y_true values
