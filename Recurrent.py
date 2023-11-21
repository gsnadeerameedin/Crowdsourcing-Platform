import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the independent variable ranges and coefficients
x_range = np.linspace(-10, 10, 50)
z_range = np.linspace(-10, 10, 50)
a = 1
b = 2

# Create a grid of x and z values
X, Z = np.meshgrid(x_range, z_range)

# Initialize y values
y_initial = np.zeros_like(X)

# Calculate y values using the recurrence relation
y_values = np.empty_like(X)
y_values[0] = y_initial
for i in range(1, len(x_range)):
    y_values[i] = y_values[i - 1] + a * x_range[i - 1] + b * z_range[i - 1]

# Create a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the solution
ax.plot_surface(X, Z, y_values, cmap='viridis')

# Set labels for axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Z-axis')
ax.set_zlabel('Y-axis')
ax.set_title('Solution of y(i+1) = y(i) + ax + bz')

# Show the plot
plt.show()
