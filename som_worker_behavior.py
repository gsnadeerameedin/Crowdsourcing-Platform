import pandas as pd
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# Load your data (replace with your data)
data = pd.read_csv('worker_behavior_data.csv')

# Select the features for clustering (replace with your features)
selected_features = ['response_time', 'Accuracy_R_Consensus']
X = data[selected_features]

# Normalize the features to the range [0, 1]
X_normalized = (X - X.min()) / (X.max() - X.min())

# Define the SOM parameters (adjust as needed)
map_width = 10
map_height = 10
input_len = len(selected_features)
sigma = 1.0
learning_rate = 0.5

# Initialize the SOM
som = MiniSom(map_width, map_height, input_len, sigma=sigma, learning_rate=learning_rate)

# Initialize the weights
som.random_weights_init(X_normalized.values)

# Train the SOM
num_epochs = 1000  # Adjust the number of epochs
som.train_batch(X_normalized.values, num_epochs)

# Find the winning neuron for each data point
winning_neurons = som.winner(X_normalized.values)

# Visualize the map
plt.figure(figsize=(map_width, map_height))
for i, (x, y) in enumerate(winning_neurons):
    plt.text(x + 0.5, y + 0.5, str(i), color=plt.cm.tab20(i // map_width / map_height), fontdict={'weight': 'bold', 'size': 9})
plt.xticks(np.arange(0, map_width))
plt.yticks(np.arange(0, map_height))
plt.grid(False)
plt.show()
