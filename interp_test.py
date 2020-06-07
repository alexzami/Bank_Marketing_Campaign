import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets,model_selection
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from ipywidgets import interactive
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from Helper import Helper
from Interpretability import Interpretability
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, get_dataset, info_plots

#def smote_tomek(x_train,y_train):
#    oversample = BorderlineSMOTE(sampling_strategy = 0.5,random_state = 0,k_neighbors = 5,m_neighbors=10,n_jobs=-1,kind = 'borderline-1')
#    X, y = oversample.fit_resample(x_train, y_train)
#    
#    tom_lin = TomekLinks(sampling_strategy='majority', n_jobs = -1)
#    X, y = tom_lin.fit_resample(X, y)

"""
Read data
"""
bank = pd.read_csv(r'data\bank-additional-full.csv', sep=';')
bank_X = bank.iloc[:,[i for i in range(0,len(bank.columns)-1)]]
bank_X = bank_X.drop('duration', axis=1) # giati to kanoume drop auto ?

target_names = ["yes", "no"]
all_columns = bank_X.columns
numerical_cols = bank_X._get_numeric_data().columns
categorial_columns = list(set(all_columns) - set(numerical_cols))

h = Helper()
interpr = Interpretability()
"""
Pre-process numerical data
"""
"""
Pre-process categorical data
"""
labelencoder_X = LabelEncoder()
for col in categorial_columns:
    bank_X[col] = labelencoder_X.fit_transform(bank_X[col])

print(bank_X)


bank_y = pd.get_dummies(bank["y"], drop_first = True)
bank_y = bank_y.rename(columns={"yes": "label"})
bank_y = bank_y["label"].values

x_train, x_test, y_train, y_test = train_test_split(bank_X, bank_y, stratify = bank_y, train_size = 0.7, random_state = 0)
#Smote - tomek links

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#----------------------------------------------------------------------------------------------------
#  WHITE BOX EXPLANATION
#----------------------------------------------------------------------------------------------------
"""
Logistic Regression
"""
lin_model = LogisticRegression(solver="newton-cg",penalty='l2',max_iter=1000,C=100,random_state=0)
lin_model.fit(x_train, y_train)
predicted_train = lin_model.predict(x_train)
y_pred = lin_model.predict(x_test)
predicted_proba_test = lin_model.predict_proba(x_test)
h.printResults(y_pred, y_test)


"""
plot the weights (coefficient of the features = feature importance)
Global interpretability
"""
feature_names = bank_X.columns.values
# interpr.plotFeaturesCoefficientGlobal(lin_model, feature_names)


"""
Local interpretation (per instance)
"""
# interpr.plotFeaturesCoefficientLocal(lin_model, x_test, y_test, y_pred, feature_names, instance=0)


#----------------------------------------------------------------------------------------------------
"""
Decision Trees
"""
model = DecisionTreeClassifier(criterion='gini',max_depth=2,random_state=0)
model.fit(x_train,y_train)
y_train_pred = model.predict(x_train)
y_pred = model.predict(x_test)
h.printResults(y_pred, y_test)

print("Decision Tree Feature importance")
print(model.feature_importances_)
#Mporoume na paiksoume me to depth
depth = 4
# interpr.plot_DecisionTree(depth, x_train, y_train, x_test, y_test, feature_names)
# interpr.plot_DecisionTree_feature_importance(depth, x_train, y_train, feature_names)




#----------------------------------------------------------------------------------------------------
#  BLACK BOX EXPLANATION
#----------------------------------------------------------------------------------------------------
# # Comments: Mathainoume ena black box model
# print("*****BLACK BOX EXPLANATION*****")
# classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
# classifier.fit(x_train, y_train)
# y_pred = classifier.predict(x_test)
# print("Random Forests Performance:")
# print(accuracy_score(y_test,y_pred))

# # Obtain predictions of the black box on the training set (could perhaps better be done via cross-validation)
# new_x_train = x_train
# new_y_train = classifier.predict(x_train)

# # epekshghsh xrhsimopoiontas dendra
# depth = 4
# # interpr.plot_DecisionTree(depth, x_train, new_y_train, x_test, y_test, feature_names, plotNamePath="results/blackbox")

# # epekshghsh xrhsimopoiontas logistic regression
# """
# Logistic Regression
# """
# lin_model = LogisticRegression(solver="newton-cg",penalty='l2',max_iter=1000,C=100,random_state=0)
# lin_model.fit(x_train, new_y_train)
# y_pred = lin_model.predict(x_test)
# predicted_proba_test = lin_model.predict_proba(x_test)
# h.printResults(y_pred, y_test)

# """
# plot the weights (coefficient of the features = feature importance)
# Global interpretability
# """
# feature_names = bank_X.columns.values
# interpr.plotFeaturesCoefficientGlobal(lin_model, feature_names)


#----------------------------------------------------------------------------------------------------
#  Feature Importance
#----------------------------------------------------------------------------------------------------
print("*****Feature Importance*****")
print("XGBoost Performance on Dataset:")
model = xgboost.XGBClassifier().fit(x_train,y_train)
y_preds = model.predict(x_test)
print(classification_report(y_test,y_preds,target_names=target_names))
perm = PermutationImportance(model).fit(x_test, y_test)
eli5.show_weights(perm, feature_names = feature_names) #gia Jupyter
print(eli5.format_as_text(eli5.explain_weights(model, feature_names=feature_names)))

#----------------------------------------------------------------------------------------------------
#  Partial Dependency Plot
#----------------------------------------------------------------------------------------------------
#TODO: Na link: https://www.kaggle.com/dansbecker/partial-plots 
# epishs na psaksw an prepei na exoun proepeksergastei ta dedomena h' oxi

print("*****Partial Dependency Plot*****")
#me ton XGBoost exei thematakia kai thelei ligo psaksimo
model = RandomForestClassifier(random_state=0, n_estimators=1000).fit(x_train, y_train)
# Create the data that we will plot
index = ['Row'+str(i) for i in range(1, len(x_test)+1)]
pdp_goals = pdp.pdp_isolate(model=model, dataset=bank_X, model_features=feature_names, feature='euribor3m')

# plot it
pdp.pdp_plot(pdp_goals, 'euribor3m')
plt.show()


#TODO: na dw auto me tous nearesNeighbours