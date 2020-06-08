# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:43:21 2020

@author: zisis
"""
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from costcla.models import BayesMinimumRiskClassifier
from costcla.metrics import cost_loss
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


bank = pd.read_csv('Desktop/ΠΜΣ ΤΝ/MACHINE LEARNING II/bank-additional-full.csv',sep = ';')
bank_X = bank.iloc[:,[i for i in range(0,len(bank.columns)-1)]]
# bank_y = bank.iloc[:,-1]
bank_X = bank_X.drop('duration', axis=1) # giati to kanoume drop auto ?

target_names = ["yes", "no"]
all_columns = bank_X.columns
numerical_cols = bank_X._get_numeric_data().columns
categorial_columns = list(set(all_columns) - set(numerical_cols))
bank_X.job.value_counts()

labelencoder_X = LabelEncoder()
for col in categorial_columns:
    bank_X[col] = labelencoder_X.fit_transform(bank_X[col])

bank_y = pd.get_dummies(bank["y"], drop_first = True)
bank_y = bank_y.rename(columns={"yes": "label"})
bank_y = bank_y["label"].values


X_train, X_test, y_train, y_test = train_test_split(bank_X, bank_y, test_size=0.3, random_state=0)



fp = np.full((y_test.shape[0],1), 1)
fn = np.full((y_test.shape[0],1), 10)
tp = np.zeros((y_test.shape[0],1))
tn = np.zeros((y_test.shape[0],1))
cost_matrix = np.hstack((fp, fn, tp, tn))


#------------ baeysian optimization + Calibration ---------------------

print("no cost minimization")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
model = clf.fit(X_train, y_train)
pred_test = model.predict(X_test)
print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides


print("no calibration")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
model = clf.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

print("costcla calibration on training set")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
model = clf.fit(X_train, y_train)
prob_train = model.predict_proba(X_train)
bmr = BayesMinimumRiskClassifier(calibration=True)
bmr.fit(y_train, prob_train) 
prob_test = model.predict_proba(X_test)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

print("\nsigmoid calibration")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
cc = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
model = cc.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides



print("\n isotonic calibration")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
cc = CalibratedClassifierCV(clf, method="isotonic", cv=3)
model = cc.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

#---------------------------------------------

#----------------------------------------------------------


    
# ΜΕΘΟΔΟς ΜΕ ΒΑΡΗ  
print("\nwith weights (alternative)")
clf = RandomForestClassifier(n_estimators=100, random_state=0, class_weight={0: 1, 1: 10})
#clf = SVC(kernel='linear', probability=False, C=1)
#clf = DecisionTreeClassifier()
model = clf.fit(X_train, y_train)
pred_test = model.predict(X_test)

print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides



# Roulete Sampling--------------------------
#---------------------------------------------
def roullete(x_train,y_train,x_test,rounds = 20):
    
    prob_list = []
    for i in range(rounds):
        sampler = RandomUnderSampler(sampling_strategy={0: 600, 1: 1500}, random_state=i)
        X_rs, y_rs = sampler.fit_sample(X_train, y_train)
        clf = RandomForestClassifier(random_state=0, n_estimators=100)
        model = clf.fit(X_rs, y_rs)
        prob_test = model.predict_proba(x_test)
        prob_list.append(prob_test)
    
    return prob_list
    
def output(problist):
    avg_prob = sum(problist)/len(problist)
    y_out = [1 if avg_prob[i][1] > 0.5 else 0 for i in range(len(avg_prob))]
    return y_out

prob_list = roullete(X_train, y_train,X_test)

pred_test = output(prob_list)    


print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

#---------------------------------------------------- 
#---------------------------------------------------------


# algos = ['random forest', 'linear SVM',' kernel SVC']
# algos_imp = [RandomForestClassifier(n_estimators=100, random_state=0), 
#                SVC(kernel='linear', probability=True, C=1),
#                SVC(kernel='rbf', probability=True, C=0.1,gamma = 0.01)]

# for clf, algo in zip(algos_imp,algos):
#     cc = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
#     model = cc.fit(X_train, y_train)
#     prob_test = model.predict_proba(X_test)
#     bmr = BayesMinimumRiskClassifier(calibration=False)
#     pred_test = bmr.predict(prob_test, cost_matrix)
#     #print(classification_report(y_test, pred_test))
#     loss = cost_loss(y_test, pred_test, cost_matrix)
#     print('{0} algorithm  cost result : {1}'.format(algo, loss))
#     # print("%d\n" %loss)