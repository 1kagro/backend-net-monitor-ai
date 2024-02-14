import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Function to create a dataset for LSTM
def create_dataset(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        # find the end of this pattern
        end_ix = i + n_steps
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# Assuming 'Total (b/s)' has been normalized
data = usage_over_time_df['Total (b/s)'].values

# Define the number of time steps to use for predictions
n_steps = 3

# Split into samples
X, y = create_dataset(data, n_steps)

# Reshape from [samples, timesteps] to [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=200, validation_split=0.2, verbose=1)
