#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\tools")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

# features_train = features_train[:int(len(features_train) / 100)]
# labels_train = labels_train[:int(len(labels_train) / 100)]

# t0 = time()
# clf = svm.SVC(kernel='rbf', C=10000.)
#
# t1 = time()
# clf.fit(features_train, labels_train)
# print("Training Time:", round(time() - t1, 3), "s")
#
# t1 = time()
# pred = clf.predict(features_test)
# print("Predict Time:", round(time() - t1, 3), "s")
#
# accuracy = accuracy_score(labels_test, pred)
#
# print(accuracy)
# print("Total Time:", round(time() - t0, 3), "s")
#
# print(len(pred))
#
# predictions = []
# x = 1
# for i in pred:
#     if i == 1:
#         predictions.append('Chris' + str(x))
#         x += 1
# print(len(predictions))


def svm_testing(value_of_c):
    t0 = time()
    clf = svm.SVC(kernel='rbf', C=value_of_c)

    t1 = time()
    clf.fit(features_train, labels_train)
    print("Training Time:", round(time() - t1, 3), "s")

    t1 = time()
    pred = clf.predict(features_test)
    print("Predict Time:", round(time() - t1, 3), "s")

    accuracy = accuracy_score(labels_test, pred)

    pred_ans = "Accuracy " + str(accuracy) + " when value of C is " + str(value_of_c)
    print(accuracy)
    print("Total Time:", round(time() - t0, 3), "s")
    print("Accuracy " + str(accuracy) + " when value of C is " + str(value_of_c))
    print("")
    return accuracy, pred_ans


# svm_testing(10)
# svm_testing(100)
# svm_testing(1000)
# svm_testing(10000)
# svm_testing(100000)

predictions = []

# for i in range(1, 10000):
#     if i == 2:
#         break
#     else:
#         value1, value2 = svm_testing(i)
#         predictions.append((value1, value2))
#     i *= 10

i = 1
while i <= 10000:
    value1, value2 = svm_testing(i)
    predictions.append((value1, value2))
    i *= 10

sorted_by_value = sorted(predictions, key=lambda x: x[0])
print("Sorted by value:", sorted_by_value)
#########################################################

#########################################################


# You'll be Provided similar code in the Quiz
# But the Code provided in Quiz has an Indexing issue
# The Code Below solves that issue, So use this one

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
