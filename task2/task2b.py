#{
# Intro. to Machine Learning
# Task 2
# Sub-task 2
# 
# Team: Pizzahawaii
# Jonas Gruetter, Levi Lingsch, Fabiano Sasselli
# 
# The goal of this task is to classify whether a certain patient will undergo sepsis.
# }

from re import L
import pandas as pd
import numpy as np
from sklearn import svm
import csv

# common ways to detect, according to some website
# Complete Blood Count - don't have this
# Lactate - have this
# CRP - don't have this
# PT and PTT - have this
# heart rate - have this
LABELS = ['pid','Lactate','PTT','Age','Temp','RRate','pH']

# import features and labels fzipzipziprom csv as pd dataframe
feature = pd.read_csv('ppdata/train_imputed.csv')
label = pd.read_csv('ppdata/train_label.csv')
test = pd.read_csv('ppdata/test_imputed.csv')
# print(feature.shape)

# all the data for a patient must fit in one row
feature = feature.groupby(np.arange(len(feature))//12).mean()
test = test.groupby(np.arange(len(test))//12).mean()

# set label for sepsis
sepsis = ['LABEL_Sepsis']
# number of patients
num_patients = len(label)
print(label[sepsis].values.ravel().max())


# create a data set using only those labels of interest
feature_interest = feature.filter(LABELS, axis=1)
test_interest = test.filter(LABELS, axis=1)

# use SVM to classify
support = svm.SVC(kernel = 'linear', C=1, probability = True)
support.fit(feature_interest, label[sepsis].values.ravel())

prediction = support.predict_proba(test_interest)
prediction_labels = ['LABEL_Sepsis_False','LABEL_Sepsis']
prediction = pd.DataFrame(data = prediction, columns = prediction_labels)
prediction.insert(loc=0,
          column='pid',
          value=test['pid'])
print(prediction.head)
prediction.to_csv('predictions/task2b.csv')