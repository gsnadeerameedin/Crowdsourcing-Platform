import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulated data for demonstration purposes
num_samples = 100
performance_score = np.random.uniform(0.5, 1.0, num_samples)  # Performance score (x)
bias_belief_score = np.random.uniform(0.0, 0.5, num_samples)  # Bias and belief score (z)
reputation_score = 0.8 * performance_score - 0.6 * bias_belief_score + np.random.normal(0, 0.05, num_samples)  # Reputation score (y)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(performance_score, bias_belief_score, reputation_score, c='blue', marker='o')

# Set labels and title
ax.set_xlabel('Performance Score (x)')
ax.set_ylabel('Bias and Belief Score (z)')
ax.set_zlabel('Reputation Score (y)')
ax.set_title('Relationship between Reputation, Performance, and Bias/Belief')

plt.show()
