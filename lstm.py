import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load your data (replace with your data)
data = pd.read_csv('worker_behavior_data.csv')

# Select the features for prediction (replace with your features)
selected_features = ['response_time', 'Accuracy_R_Consensus', 'Accuracy_R_Gold', 'Total_Tasks', 'completion_rate', 'ResponseTime']
X = data[selected_features].values

# Normalize features to the range [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define window size for time series data
window_size = 10  # Adjust as needed

# Prepare data for LSTM (create sequences)
X_sequence = []
y = []
for i in range(window_size, len(X_scaled)):
    X_sequence.append(X_scaled[i - window_size:i])
    y.append(X_scaled[i, 0])  # Predicting response_time

X_sequence = np.array(X_sequence)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequence, y, test_size=0.2, random_state=0)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))  # Single output neuron

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Denormalize predictions for visualization
y_pred_denorm = scaler.inverse_transform(np.column_stack((y_pred, X_test[:, 1:])))
y_test_denorm = scaler.inverse_transform(np.column_stack((y_test.reshape(-1, 1), X_test[:, 1:])))

# Plot predicted vs. actual response_time
plt.plot(y_test_denorm[:, 0], label='Actual')
plt.plot(y_pred_denorm[:, 0], label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Response Time')
plt.legend()
plt.show()
