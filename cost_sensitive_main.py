import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from costcla.models import BayesMinimumRiskClassifier
from costcla.metrics import cost_loss
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from interpretability_plots import Interpretability
# from sklearn 

# load dataset --------------------------------------------------

bank = pd.read_csv(r'.\data\bank-additional-full.csv', sep=';')
bank_X = bank.iloc[:,[i for i in range(0,len(bank.columns)-1)]]
# bank_y = bank.iloc[:,-1]
bank_X = bank_X.drop('duration', axis=1)

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

#----------------- cost matrix -------------------------------------------

fp = np.full((y_test.shape[0],1), -10)
fn = np.zeros((y_test.shape[0],1))
tp = np.full((y_test.shape[0],1),100)
tn = np.zeros((y_test.shape[0],1))
cost_matrix = np.hstack((fp, fn, tp, tn))

fp = np.full((y_train.shape[0],1), -10)
fn = np.zeros((y_train.shape[0],1))
tp = np.full((y_train.shape[0],1),100)
tn = np.zeros((y_train.shape[0],1))
cost_matrix_train = np.hstack((fp, fn, tp, tn))

#---------------------------------------------------------------------
#============= isotonic + bayesin + black box tree ---------------------

def baeysian_clas(train, test, val_trai,val_test,
                  auto_calibration = False,calibration_func = None,
                  clf = None, CostMatrix = None, CostMatrixTrain = None ):
    
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    val_trai = scaler.fit_transform(val_trai)
    
    if calibration_func is None:
        model = clf.fit(train, test)
    else :
        cc = CalibratedClassifierCV(clf, method= calibration_func, cv=3)
        model = cc.fit(train, test)


    prob_test = model.predict_proba(val_trai)
    bmr = BayesMinimumRiskClassifier(calibration= auto_calibration)
    pred_test = bmr.predict(prob_test, CostMatrix)
    
    prob_test_train = model.predict_proba(train)
    bmr_train = BayesMinimumRiskClassifier(calibration= auto_calibration)
    pred_train = bmr_train.predict(prob_test_train, CostMatrixTrain)

    print(classification_report(val_test, pred_test))
    loss = cost_loss(val_test, pred_test, CostMatrix)
    print("%d\n" %loss)
    print(confusion_matrix(val_test, pred_test).T)
    return pred_train , pred_test

svm = SVC(kernel = 'rbf', C = 0.1, gamma = 0.1 , probability = True)

# baeysian_clas(X_train,y_train, X_test,y_test, clf = svm,
#                    calibration_func = 'isotonic', CostMatrix = cost_matrix)

# baeysian_clas(X_train,y_train, X_test,y_test, clf = svm,
#                calibration_func = None, CostMatrix = cost_matrix,
#                auto_calibration = False)

new_x_train = X_train
new_y_train ,test_pred = baeysian_clas(X_train, y_train, X_test, y_test, clf = svm,
                   calibration_func = 'isotonic', CostMatrix = cost_matrix,
                   CostMatrixTrain = cost_matrix_train )

iterp = Interpretability()
iterp.plot_DecisionTree(3,new_x_train, new_y_train, test_pred,X_test, y_test,all_columns)







# Roulete Sampling + black box --------------------------
#---------------------------------------------
def roullete(x_train,y_train,x_test,rounds = 20):
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)    
    
    prob_list = []
    for i in range(rounds):
        sampler = RandomUnderSampler(sampling_strategy={0: 300, 1: 3000}, random_state=i)
        X_rs, y_rs = sampler.fit_sample(x_train, y_train)
        clf = SVC(kernel = 'rbf', C = 0.1, gamma = 0.1 , probability = True)
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

new_x_train = X_train
prob_list_train = roullete(X_train, y_train,X_train)
new_y_train = output(prob_list_train)


print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides


iterp = Interpretability()
iterp.plot_DecisionTree(3,new_x_train, new_y_train, pred_test,X_test, y_test,all_columns)









    
# weight method + tree feature importance. ...................................


print("\nwith weights (alternative)")
clf = LogisticRegression(max_iter = 1000, random_state=0, class_weight={0: 1, 1: 10})
#clf = SVC(kernel='linear', probability=False, C=1)
#clf = DecisionTreeClassifier()
model = clf.fit(X_train, y_train)
pred_test = model.predict(X_test)

print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides




feature_names = bank_X.columns.values
interpr = Interpretability()
interpr.plotFeaturesCoefficientGlobal(clf, feature_names)


#------------------- weighted decision tree ---------------



from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=0,
                               class_weight={0: 1, 1: 10})
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
# h.printResults(y_pred, y_test)
print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, y_pred, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, y_pred).T) # transpose to align with slides



print("Decision Tree Feature importance")
print(model.feature_importances_)
#Mporoume na paiksoume me to depth
depth = 3
interpr = Interpretability()
interpr.plot_DecisionTree(depth, X_train, y_train,y_pred, X_test, y_test, all_columns)
interpr.plot_DecisionTree_feature_importance(depth, X_train, y_train, feature_names)

#---------------------------------------------------------------------------

#--------- GRID SEARCH -----------------------


fp = np.full((y_test.shape[0],1), -3)
fn = np.zeros((y_test.shape[0],1))
tp = np.full((y_test.shape[0],1),200)
tn = np.zeros((y_test.shape[0],1))
cost_matrix_cv = np.hstack((fp, fn, tp, tn))[:2621]

def cost_loss_score(y_test, pred_test):
    return cost_loss(y_test, pred_test, cost_matrix_cv)

my_score = make_scorer(cost_loss_score)

scaler = MinMaxScaler()
svm = SVC(kernel = 'rbf')
logmodel = LogisticRegression(max_iter = 600) 
clf = RandomForestClassifier(random_state=0)
pipeline = Pipeline(steps = [('scaler',scaler), ('classifier',svm)])

# param_grid = {'classifier__n_estimators' : [10,50,100,200,300 ],
#               'classifier__criterion' : ['gini', 'entropy'],
#               'classifier__max_depth' : [5,10,20]}

# param_grid_log = {'classifier__C' : [0.01,0.1,1,10,100,1000],
#               'classifier__max_iter' : [600, 1200]}

param_grid_svm = {'classifier__gamma' : [0.001,0.01,0.1],
              'classifier__C' : [0.1, 1]}

grid_search  = GridSearchCV(pipeline, param_grid=param_grid_svm, cv = 11, scoring=my_score, n_jobs=-1)
grid_search.fit(X_train,y_train)

results = grid_search.cv_results_

#...........................................................







