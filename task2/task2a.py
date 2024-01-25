#{
# Intro. to Machine Learning
# Task 2
# Sub-task 1
# 
# Team: Pizzahawaii
# Jonas Gruetter, Levi Lingsch, Fabiano Sasselli
# 
# The goal of this task is to determine whether further tests are necessary
# for a certain patient. This code should produce probabilistic labels using
# the ROC AUC. 
# }

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn as sk
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


## if certain tests have already been completed, it is likely not necessary to 
## conduct further tests. At least this is what I am thinking.

# import features and labels from csv as pd dataframe
feature = pd.read_csv('ppdata/train_imputed.csv')
label = pd.read_csv('ppdata/train_label.csv')
test = pd.read_csv('ppdata/test_imputed.csv')

###

# use sklearn's binary classification capabilities
# trying svm
# SVM = svm.LinearSVC(max_iter=10000)

TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

y_train_1 = label[TESTS]
num_patients = len(label)

# using SVM on each column
# model = svm.LinearSVC(max_iter=10000)
col = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

# for all drugs, consider a table where row = pid, column = hour
y_pred = np.zeros([num_patients, 10])

k=0
for label in TESTS:
    corr_train_label = label[6:]

    # First reshape the data such that the X_training and X_test are in R^{n_sample,n_features}
    #training data
    train_considered = feature[corr_train_label].to_numpy()
    x_label_np_train = np.zeros((num_patients, 12))
    # Testing data
    test_considered = feature[corr_train_label].to_numpy()
    x_label_np_test = np.zeros((num_patients, 12))
    for i in range(num_patients):
        for j in range(12):
            x_label_np_train[i,j] = train_considered[i+j]
            x_label_np_test[i,j] = test_considered[i+j]
    
    # Put X as pd DataFrame
    x_label_train = pd.DataFrame(x_label_np_train, columns = col)
    y_label_train = y_train_1[label]
    
    x_label_test = pd.DataFrame(x_label_np_test, columns = col)
    
    # Training (Logistic regression with cross validation)
    model = LogisticRegressionCV(dual=False, class_weight='balanced', cv=2, random_state=0, penalty='l2', max_iter=10000, solver='saga')
    model.fit(x_label_train,y_label_train)
    y_pred[:,k] = model.predict_proba(x_label_test)[:,1]
    #print(model.score(x_label_train, y_label_train))
    k = k+1

y_pd = pd.DataFrame(data = y_pred, 
                  columns = labels_1)

y_pd.to_csv('task1.csv')
task1 = pd.read_csv('task1.csv').iloc[:,1:]
