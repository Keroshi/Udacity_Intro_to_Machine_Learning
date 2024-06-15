#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\tools")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score
from time import time

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print(len(features_train[0]))

#########################################################
### your code goes here ###

t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(pred)
accuracy = accuracy_score(labels_test, pred)
print(accuracy)
print(f"Total Time: {round(time() - t0, 3)} s")

#########################################################
