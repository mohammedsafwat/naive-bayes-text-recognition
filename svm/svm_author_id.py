#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#use 1% of the training data set to speed up computations
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]



#########################################################
### your code goes here ###
clf = svm.SVC(kernel='rbf', C=10000)
clf.fit(features_train, labels_train)

t0 = time()
pred = clf.predict(features_test)
print("prediction time = " + str(round(time() - t0, 3)) + "s")

accuracy = accuracy_score(pred, labels_test)
print("accuracy = " + str(accuracy))

#if pred[i] where i is the index of the element equals to 0 then it's Sara
#if it's equal to 1 then it's Chris
pred_chris_indices = []
for i in range(0, len(pred)):
    if pred[i] == 1:
       pred_chris_indices.append(i)

print("number of emails sent by chris in" + str(len(pred)) + " test cases = " + str(len(pred_chris_indices)))

#########################################################


