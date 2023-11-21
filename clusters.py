import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Simulated data for demonstration purposes
num_samples = 100
performance_score = np.random.uniform(0.5, 1.0, num_samples)
bias_belief_score = np.random.uniform(0.0, 0.5, num_samples)
reputation_score = 0.8 * performance_score - 0.6 * bias_belief_score + np.random.normal(0, 0.05, num_samples)

# Combine the scores into a feature matrix
X = np.column_stack((performance_score, bias_belief_score, reputation_score))

# Apply K-Means clustering with 3 clusters
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
labels = kmeans.fit_predict(X)

# Create a 3D scatter plot with colored clusters
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(performance_score, bias_belief_score, reputation_score, c=labels, marker='o')

# Add a colorbar
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)

# Set labels and title
ax.set_xlabel('Performance Score (x)')
ax.set_ylabel('Bias and Belief Score (z)')
ax.set_zlabel('Reputation Score (y)')
ax.set_title('Clusters of Data Points with K-Means')

plt.show()
