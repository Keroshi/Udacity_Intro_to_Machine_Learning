import numpy as np
import matplotlib.pyplot as plt

# Correct data for plotting
X = np.array([-1, -2, -3, 1, 2, 3])
Y = np.array([-1, -1, -2, 1, 1, 2])

# Fit a linear trend line to the data
# coefficients = np.polyfit(X, Y, 1)
# trend_line = np.poly1d(coefficients)

# Plot the data points
plt.scatter(X, Y, color='blue', label='Data points')

# Plot the trend line
# x_vals = np.linspace(min(X), max(X), 100)
# plt.plot(x_vals, trend_line(x_vals), color='red', label='Trend line')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Data points with Trend Line')
plt.legend()

# Show the plot
plt.show()
