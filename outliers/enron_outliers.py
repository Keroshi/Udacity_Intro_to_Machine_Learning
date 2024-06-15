#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot

sys.path.append(os.path.abspath("C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\tools"))
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = joblib.load(open(
    "C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\final_project\\final_project_dataset.pkl",
    "rb"))
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

### your code below
feature_selection = []

print(data)

for i in data_dict:
    if data_dict[i]["salary"] != 'NaN' and data_dict[i]["bonus"] != 'NaN':
        if data_dict[i]["salary"] != 0 and data_dict[i]["bonus"] != 0:
            feature_selection.append([data_dict[i]["salary"], data_dict[i]["bonus"]])

for i in data_dict:
    if data_dict[i]["salary"] != "NaN" and data_dict[i]["bonus"] != "NaN":
        if int(data_dict[i]["salary"]) >= 1000000 and int(data_dict[i]["bonus"]) >= 1000000:
            print(i)

for i in data_dict:
    for j in data_dict[i]:
        if data_dict[i][j] == 6680544:
            print(f"{data_dict[i]}")

min_max = []

for i in data_dict:
    if data_dict[i]['salary'] != 'NaN':
        min_max.append(data_dict[i]['salary'])

min_max_sorted = sorted(min_max)
print(min_max_sorted)

for i in data_dict:
    if data_dict[i]['salary'] == 1060932:
        print(i)
    elif data_dict[i]['exercised_stock_options'] == 1000000:
        print(i)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

feature_selection = np.array(feature_selection)

X = feature_selection[:, 0]
y = feature_selection[:, 1]

# X = data[:, 0]  # salary
# y = data[:, 1]  # bonus

# Reshape if necessary
X = X.reshape(-1, 1)
print(X)
print(y)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Print the coefficients
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

# Plot the results
plt.scatter(X, y, color='blue')
plt.plot(X, reg.predict(X), color='red', linewidth=2)
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.title('Salary vs Bonus Regression')
plt.show()
