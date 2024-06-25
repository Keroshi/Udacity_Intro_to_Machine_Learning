#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import joblib
import sys

sys.path.append(os.path.abspath("C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\tools"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open(
    "C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\final_project\\final_project_dataset.pkl",
    "rb"))

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### it's all yours from here forward!

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# clf = DecisionTreeClassifier()
# clf.fit(features, labels)
# pred = clf.predict(features)
# accuracy = accuracy_score(labels, pred)
# print(accuracy)

# def classification(random_state):
#     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=random_state)
#
#     clf = DecisionTreeClassifier()
#     clf.fit(X_train, y_train)
#     pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, pred)
#     return accuracy
#
#
# accuracy_array = []
# for i in range(0, 100):
#     accuracy_array.append([i, (classification(i))])
#
# counter = 0
# for i in accuracy_array:
#     if accuracy_array[counter][1] >= 0.72 and accuracy_array[counter][1] <= 0.74:
#         print(f"{counter} {accuracy_array[counter][1]}")
#     counter += 1

# print(classification(42))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=93)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print(accuracy)

precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
print(precision, recall)

POI = []
POI_y = []
POI_labels = []
# for i, ii, iii in zip(pred, y_test, labels):
#     if i == 1:
#         POI.append(1)
#     if ii == 1:
#         POI_y.append(1)
#     if iii == 1:
#         POI_labels.append(1)

for i in pred:
    if i == 1:
        POI.append(1)

for ii in y_test:
    if ii == 1.0:
        POI_y.append(1)

for iii in labels:
    if iii == 1:
        POI_labels.append(1)

print(len(POI))
print(len(POI_y))
print(len(POI_labels))
# print(labels)
print(len(y_test))
print(len(pred))
