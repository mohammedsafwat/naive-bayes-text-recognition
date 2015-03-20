#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
#classifier
clf = GaussianNB()

t0 = time()
#fit it
clf.fit(features_train, labels_train)
print("training time = " + str( round(time() - t0, 3) ) + "s")

t1 = time()
#vector of predictions
pred = clf.predict(features_test)
print("prediction time = " + str(round(time() - t1, 3)) + "s")
#accuracy         naive_bayes/nb_author_id.py:33
#number of points classified correctly / number of total points of test set
accuracy = accuracy_score(pred, labels_test)
print("accuracy = " + str(accuracy))

#########################################################


