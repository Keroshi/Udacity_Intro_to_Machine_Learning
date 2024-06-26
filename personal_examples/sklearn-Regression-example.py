from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

# Prepare data for plotting
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]

# Fit the model
reg = LinearRegression()
reg.fit(X, y)

# Extract coefficients
print(reg.coef_)

accuracy = r2_score(y, reg.predict(X))
print(accuracy)

# Plotting
plt.scatter([x[0] for x in X], y, color='blue')  # Plot the data points
plt.plot([x[0] for x in X], reg.predict(X), color='red')  # Plot the regression line
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Fit')
plt.show()
