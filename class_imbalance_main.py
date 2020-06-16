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
# from sklearn.pipeline import Pipeline
import imblearn
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks,NearMiss
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn import svm
from graphviz import Source
from Interpretability import Interpretability
from Helper import Helper
from sklearn.metrics import accuracy_score
#%matplotlib qt

interpr = Interpretability()
h = Helper()

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


bank = pd.read_csv(r'.\data\bank-additional-full.csv', sep=';')
bank_X = bank.iloc[:,[i for i in range(0,len(bank.columns)-1)]]
# bank_y = bank.iloc[:,-1]
bank_X = bank_X.drop('duration', axis=1) # to kanoume drop giati einai cheat

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

x_train, x_test, y_train, y_test = train_test_split(bank_X, bank_y, stratify = bank_y, train_size = 0.7, random_state = 0)

#classifier = svm.SVC(kernel = 'rbf',C=1000,gamma=0.001) 
classifier = LogisticRegression(max_iter = 10000, C = 0.1) 

#easy_ensemble = imblearn.ensemble.EasyEnsembleClassifier(n_estimators=35, base_estimator=classifier, sampling_strategy='majority', n_jobs=-1)

oversample = BorderlineSMOTE(sampling_strategy = 0.5,n_jobs=-1, kind = 'borderline-1')
x_train, y_train = oversample.fit_resample(x_train, y_train)
tom_lin = TomekLinks(sampling_strategy = 'majority',n_jobs = -1)
x_train, y_train = tom_lin.fit_resample(x_train, y_train)


classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)


h.printResults2(y_test, y_pred)
h.plotConfusionMatrix(y_test, y_pred, norm=True)
h.plotConfusionMatrix(y_test, y_pred, norm=False)

#White-box explanation
feature_names = bank_X.columns.values
interpr.plotFeaturesCoefficientGlobal(classifier, feature_names)


new_x_train = x_train
new_y_train = classifier.predict(x_train)

feature_names = bank_X.columns.values

depth = 3
interpr.plot_DecisionTree(depth, new_x_train, new_y_train, y_pred,  x_test, y_test, feature_names)
interpr.plot_DecisionTree_feature_importance(depth, new_x_train, new_y_train, feature_names)

"""
Logistic Regression
"""
lin_model = LogisticRegression(solver="newton-cg",penalty='l2',max_iter=1000,C=100,random_state=0)
lin_model.fit(new_x_train, new_y_train)
y_pred_lin = lin_model.predict(x_test)
print("Fidelity",accuracy_score(y_pred, lin_model.predict(x_test)))
h.printResults2(y_pred_lin, y_test)

"""
plot the weights (coefficient of the features = feature importance)
Global interpretability
"""
feature_names = bank_X.columns.values
interpr.plotFeaturesCoefficientGlobal(lin_model, feature_names)
