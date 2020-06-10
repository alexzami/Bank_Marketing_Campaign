'''
Created on 10 May 2020

@author: 30693
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
import imblearn
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks,NearMiss

def smote_tomek(x_train,y_train):
    oversample = BorderlineSMOTE(sampling_strategy = 0.5, random_state = 0, k_neighbors = 5, m_neighbors=10, n_jobs=-1, kind = 'borderline-1')
    X, y = oversample.fit_resample(x_train, y_train)
    
    tom_lin = TomekLinks(sampling_strategy='majority', n_jobs = -1)
    X, y = tom_lin.fit_resample(X, y)
    # print(len([i for i in y_train.values if i==1]))
    # print(len([i for i in y.values if i==1]))
    # print(len(y_train))
    # print(len(y))
    return X,y    

def easy_ensemble_clasiffication(classifier):
    
    easy_ensemble = imblearn.ensemble.EasyEnsembleClassifier(n_estimators=3, base_estimator=classifier, sampling_strategy='majority', n_jobs=-1)
    return(easy_ensemble)

def near_miss (x_train,y_train):
    
    nm = NearMiss (sampling_strategy = 'majority',n_neighbors = 5,version=1)
    X, y = nm.fit_resample(x_train, y_train)
    
    return X, y

bank = pd.read_csv(r'data\bank-additional-full.csv', sep=';')
bank_X = bank.iloc[:,[i for i in range(0,len(bank.columns)-1)]]
bank_X = bank_X.drop('duration', axis=1)
bank_y = bank.iloc[:,-1]


all_columns = bank_X.columns
numerical_cols = bank_X._get_numeric_data().columns
categorial_columns = list(set(all_columns) - set(numerical_cols))

for cat in categorial_columns:
    bank_X = pd.concat([bank_X.drop(cat, axis=1), pd.get_dummies(bank_X[cat],prefix=cat, drop_first = True)], axis=1)
 
bank_y = pd.get_dummies(bank_y, prefix = ['y'], drop_first = True)

x_train, x_test, y_train, y_test = train_test_split(bank_X, bank_y, stratify = bank_y, train_size = 0.7, random_state = 0)



#Smote - tomek links
x_train,y_train = smote_tomek(x_train,y_train)

#Easy Ensemble
# logmodel = LogisticRegression(max_iter = 600) 
# easy_ensemble = easy_ensemble_clasiffication()
#Near Miss
# x_train,y_train = near_miss(x_train,y_train)

scaler = MinMaxScaler()
    
logmodel = LogisticRegression(max_iter = 600) 
pipeline = Pipeline(steps = [('scaler',scaler), ('classifier',logmodel)])

param_grid = {'scaler__feature_range':[(0,1)]}

grid_search  = GridSearchCV(pipeline, param_grid=param_grid, cv = 10, scoring="f1", n_jobs=-1, iid = False, verbose=1)
grid_search.fit(x_train,y_train.values.ravel())
y_pred = grid_search.predict(x_test)



print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.refit_time_)


accuracy = metrics.accuracy_score(y_test.values.ravel(),y_pred)
precision = metrics.precision_score(y_test.values.ravel(),y_pred)
recall = metrics.recall_score(y_test.values.ravel(),y_pred)
f1 = metrics.f1_score(y_test.values.ravel(),y_pred)
geometric_mean = imblearn.metrics.geometric_mean_score(y_test.values.ravel(),y_pred)
balanced_accuracy = metrics.balanced_accuracy_score(y_test.values.ravel(),y_pred) 
 
 
print("Recall: "+str(round((recall*100),2))+"%")
print("Precision: "+str(round((precision*100),2))+"%")
print("Accuracy: "+str(round((accuracy*100),2))+"%")
print("F1: "+str(round((f1*100),2))+"%")
print("Geometric Mean: "+str(round((geometric_mean*100),2))+"%")
print("Balanced Accuracy: "+str(round((balanced_accuracy*100),2))+"%")
