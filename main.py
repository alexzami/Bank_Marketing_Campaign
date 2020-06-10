import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Helper import Helper
from Interpretability import Interpretability
from sklearn.linear_model import LogisticRegression
from ClassImbalance import ClassImbalance
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from costcla.models import BayesMinimumRiskClassifier
from sklearn.metrics import confusion_matrix, classification_report
from costcla.metrics import cost_loss



#----------------------------------------------------------------------------------------------------
#  Read & pre-process data
#----------------------------------------------------------------------------------------------------
bank = pd.read_csv(r'data\bank-additional-full.csv', sep=';')
bank_X = bank.iloc[:,[i for i in range(0,len(bank.columns)-1)]]
# bank_y = bank.iloc[:,-1]
bank_X = bank_X.drop('duration', axis=1) # to kanoume drop giati einai cheat

target_names = ["yes", "no"]
all_columns = bank_X.columns
numerical_cols = bank_X._get_numeric_data().columns
categorial_columns = list(set(all_columns) - set(numerical_cols))
bank_X.job.value_counts()

# TODO: na dokimasoume one hot encoding
labelencoder_X = LabelEncoder()
for col in categorial_columns:
    bank_X[col] = labelencoder_X.fit_transform(bank_X[col])

bank_y = pd.get_dummies(bank["y"], drop_first = True)
bank_y = bank_y.rename(columns={"yes": "label"})
bank_y = bank_y["label"].values

x_train, x_test, y_train, y_test = train_test_split(bank_X, bank_y, stratify = bank_y, train_size = 0.7, random_state = 0)

h = Helper()
interpr = Interpretability()
classImb = ClassImbalance()

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#----------------------------------------------------------------------------------------------------
#  BLACK BOX EXPLANATION
#----------------------------------------------------------------------------------------------------

def create_cost_matrix(y_test):
  fp = np.full((y_test.shape[0],1), 1)
  fn = np.full((y_test.shape[0],1), 10)
  tp = np.zeros((y_test.shape[0],1))
  tn = np.zeros((y_test.shape[0],1))
  cost_matrix = np.hstack((fp, fn, tp, tn))
  return cost_matrix
"""
Logistic Regression
"""
cost_matrix = create_cost_matrix(y_test)

logmodel = LogisticRegression(solver="newton-cg",penalty='l2',max_iter=1000,C=100,random_state=0) 
x_train, y_train = classImb.smote_tomek(x_train, y_train)
logmodel = CalibratedClassifierCV(logmodel, method="sigmoid", cv=3)
logmodel.fit(x_train, y_train)
prob_test = logmodel.predict_proba(x_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

#predicted_train = logmodel.predict(x_train)
#y_pred = logmodel.predict(x_test)
#h.printResults(y_pred, y_test)

#feature_names = bank_X.columns.values
#interpr.plotFeaturesCoefficientGlobal(logmodel, feature_names)