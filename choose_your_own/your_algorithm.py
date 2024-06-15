#!/usr/bin/python
import sys

sys.path.append('C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\choose_your_own')
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# # Using Naive Bayes Algorithm
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
#
# t0 = time()
# clf = GaussianNB()
# clf.fit(features_train, labels_train)
# y_pred = clf.predict(features_test)
# accuracy = accuracy_score(labels_test, y_pred)
# print(f'This is the accuracy score for Naive Bayes Algorithm: {accuracy * 100}%')
# print(f'Total Time: {round(time() - t0, 3)} s')

# # Using SVM Classifier Algorithm
# from sklearn import svm
# from sklearn.metrics import accuracy_score
#
# t0 = time()
# clf = svm.SVC(kernel='rbf', gamma=1.0, C=1.0)
# clf.fit(features_train, labels_train)
# y_pred = clf.predict(features_test)
# accuracy = accuracy_score(labels_test, y_pred)
# print(f'This is the accuracy score for SVM Algorithm: {accuracy * 100}%')
# print(f'Total Time: {round(time() - t0, 3)} s')

# # Using Decision Tree Classifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
#
# t0 = time()
# clf = DecisionTreeClassifier(criterion="gini", min_samples_split=20)
# clf.fit(features_train, labels_train)
# y_pred = clf.predict(features_test)
# accuracy = accuracy_score(labels_test, y_pred)
# print(f'This is the accuracy score for the Decision Tree Classifier Algorithm: {accuracy * 100}%')
# print(f'Total Time: {round(time() - t0, 3)} s')

# Using K Nearest Neighbours Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

t0 = time()
clf = KNeighborsClassifier(n_neighbors=18)
clf.fit(features_train, labels_train)
y_pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, y_pred)
print(f'This is the accuracy score for K Nearest Neighbours Classifier Algorithm: {accuracy * 100}%')
print(f'Total Time: {round(time() - t0, 3)} s')

# # Using Random Forest Classifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
#
# t0 = time()
# clf = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=0, min_samples_split=20)
# clf.fit(features_train, labels_train)
# y_pred = clf.predict(features_test)
# accuracy = accuracy_score(labels_test, y_pred)
# print(f'This is the accuracy score for the Random Forest Classifier Algorithm: {accuracy * 100}%')
# print(f'Total Time: {round(time() - t0, 3)} s')

# # Using Adaboost Classifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import accuracy_score
#
# t0 = time()
# clf = AdaBoostClassifier(n_estimators=16, random_state=0, algorithm='SAMME')
# clf.fit(features_train, labels_train)
# y_pred = clf.predict(features_test)
# accuracy = accuracy_score(labels_test, y_pred)
# print(f'This is the accuracy score for the Adaboost Algorithm: {accuracy * 100}%')
# print(f'Total Time: {round(time() - t0, 3)}s')
# print(f'NUmber of features: {len(features_train)}')

# # Using Adaboost Classifier with Random Forest Classifier as the model
# from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from sklearn.metrics import accuracy_score
#
# t0 = time()
# model = RandomForestClassifier(n_estimators=1000, criterion='gini', random_state=0, min_samples_split=20)
# clf = AdaBoostClassifier(estimator=model, n_estimators=7, random_state=0, algorithm='SAMME')
# clf.fit(features_train, labels_train)
# y_pred = clf.predict(features_test)
# accuracy = accuracy_score(labels_test, y_pred)
# print(
#     f'THis is the accuracy score for the Adaboost Classifier Algorithm with Random Forest Algorithm as a model : {accuracy * 100}%')
# print(f'Total Time: {round(time() - t0, 3)} s')

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    print("Error")
    pass
