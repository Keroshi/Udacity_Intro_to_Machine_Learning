#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys

sys.path.append(os.path.abspath("C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\tools"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open(
    "C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\final_project\\final_project_dataset.pkl",
    "rb"))

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)
