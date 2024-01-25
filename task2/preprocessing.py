#{
# Intro. to Machine Learning
# Task 2
# Sub-task 1
# 
# Team: Pizzahawaii
# Jonas Gruetter, Levi Lingsch, Fabiano Sasselli
# 
# The goal of this code is to preprocess the data so that we may work with
# it more easily for the rest of the subtasks.
# }

# import pandas and numpy 
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import csv

###
# import and sort the data

# import features and labels from csv as pd dataframe
train_feature = pd.read_csv('train_features.csv')
train_label = pd.read_csv('train_labels.csv')
test_feature = pd.read_csv('test_features.csv')
sample = pd.read_csv('sample.csv')

# sort the values by pid, and then time to make it easier to read
train_feature.sort_values(by = ['pid', 'Time'], inplace = True)
train_label.sort_values(by = 'pid', inplace = True)
test_feature.sort_values(by = ['pid', 'Time'], inplace = True)
sample.sort_values(by = ['pid'], inplace = True)


# first we want to interpolate between values that have already been entered for a given pid
# training data
for i in range(0,train_feature.shape[0],12):
    train_feature.iloc[i:i+12, :] = train_feature.iloc[i:i+12, :].interpolate(limit_direction='both', axis = 0)
# print(train_feature.head(20))  

# test data
for i in range(0,test_feature.shape[0],12):
    test_feature.iloc[i:i+12, :] = test_feature.iloc[i:i+12, :].interpolate(limit_direction='both', axis = 0)
# print(train_feature.head(20))  


# impute values within the data
# use sklearn's imputer capabilities
# can use 'median' or 'mean'
fill_NaN = SimpleImputer(missing_values=np.nan, strategy='median')
train_imp = pd.DataFrame(fill_NaN.fit_transform(train_feature))
train_imp.columns = train_feature.columns
train_imp.index = train_feature.index

test_imp = pd.DataFrame(fill_NaN.fit_transform(test_feature))
test_imp.columns = test_feature.columns
test_imp.index = test_feature.index



# normalize the data using mean normalization
train_pid = train_imp.pop('pid')
train_imp = (train_imp - train_imp.mean())/train_imp.std()
train_imp.insert(1, 'pid', train_imp)

# test_pid  = test_imp.pop('pid')
# test_imp  = (test_imp - test_imp.mean())/test_imp.std()
# test_imp.insert(1, 'pid', test_imp)
# print(test_imp.head(20))



# write the preprocessed data to new files for speed
train_imp.to_csv('ppdata/train_imputed.csv')
test_imp.to_csv('ppdata/test_imputed.csv')
train_label.to_csv('ppdata/train_label.csv')
sample.to_csv('ppdata/sample.csv')
