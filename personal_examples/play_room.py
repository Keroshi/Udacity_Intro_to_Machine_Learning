# from matplotlib import pyplot as plt
# from sklearn.linear_model import LinearRegression
#
# X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# y = [1, 2, 3]
#
# reg = LinearRegression()
# reg.fit(X, y)
#
# print(reg.coef_)
# print(reg.intercept_)
#
# plt.scatter([x[0] for x in X], y)
# plt.plot([x[0] for x in X], reg.predict(X))
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Play Room Regression')
# plt.show()


# from matplotlib import pyplot as plt
# from sklearn.linear_model import LinearRegression
# import numpy as np
#
# X = np.array([1, 2, 3]).reshape(-1, 1)
# y = np.array([1, 2, 3])
#
# reg = LinearRegression()
# reg.fit(X, y)
#
# # X_feature_1 = X[:, 0].reshape(-1, 1)
#
#
# print(reg.coef_)
# print(reg.intercept_)
#
# plt.scatter(X, y, color='blue', label='Data points')
# plt.plot(X, reg.predict(X), color='red', label='Regression line')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Play Room Regression')
# plt.legend()
# plt.show()

# def feature_scaling(arr):
#     arr_sorted = sorted(arr)
#     x_prime = ((arr_sorted[1] - arr_sorted[0]) / (arr_sorted[2] - arr_sorted[0]))
#
#     return x_prime
#
#
# data = [115, 140, 175]
# feature_scaling(data)

# def featureScaling(arr):
#     x_min = min(arr)
#     x_max = max(arr)
#
#     if x_min == x_max:
#         return [0.5] * len(arr)
#
#     # return [(round(((x - x_min) / (x_max - x_min)) * 2) / 2) for x in arr]
#     return [(x - x_min) / (x_max - x_min) for x in arr]
#
#
# data = [115, 140, 175]
# featureScaling(data)

# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score
# from matplotlib import pyplot as plt
# from time import time
#
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])
#
# print(X.shape)
# print(Y.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
#
#
# def classifier(n_estimators):
#     clf = RandomForestClassifier(n_estimators=n_estimators, random_state=24)
#     clf.fit(X_train, y_train)
#     pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, pred)
#     return accuracy
#
#
# plot_points_x = []
# plot_points_y = []
#
# t0 = time()
#
# for x in range(1, 101):
#     classifier(x)
#     plot_points_x.append(x)
#     plot_points_y.append(classifier(x))
#
# print(f"Total time: {round(time() - t0, 2)}s")
#
# plt.scatter(plot_points_x, plot_points_y)
# plt.xlabel("n_estimators")
# plt.ylabel("accuracy")
# plt.title("Plot for n_estimators against accuracy")
# plt.show()

# from sklearn.metrics import precision_score, recall_score
#
# true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
#
# precision = precision_score(true_labels, predictions)
# recall = recall_score(true_labels, predictions)
# print(precision, recall)
#
# combined_array = []
# counter = 0
# for i in true_labels:
#     combined_array.append([true_labels[counter], predictions[counter]])
#     counter += 1
#
# true = []
# counter = 0
# for i in combined_array:
#     if combined_array[counter][0] == combined_array[counter][1]:
#         if combined_array[counter][0] == 1 and combined_array[counter][1] == 1:
#             true.append(1)
#     counter += 1
# print(combined_array)
# print(len(true))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate data for the sine wave
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Create a figure and an axis
fig, ax = plt.subplots()
line, = ax.plot(x, y)

# Set the axes limits
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.1, 1.1)


# Define the animation function
def animate(i):
    line.set_ydata(np.sin(x + i / 10.0))
    return line,


# Create the animation object
ani = animation.FuncAnimation(fig, animate, frames=100, interval=20, blit=True)

# Display the animation
plt.show()

# Optionally save the animation as a GIF or MP4
# ani.save('sine_wave_animation.mp4', writer='ffmpeg')
# ani.save('sine_wave_animation.gif', writer='imagemagick')
