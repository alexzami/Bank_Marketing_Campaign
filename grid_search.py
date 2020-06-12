import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%matplotlib qt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
import pickle
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import datetime as dt

#------------------Normalize data----------------------#
X2 = X/255.0
x_test = x_test/255.0


# Gia na kanw to gridSearch gia kernel poly kai rbf evala/evgala antistoixa apo sxolia 
# tis parametrous pou xreiazotan kathe fora
kpca = KernelPCA()
lda = LDA()
knn = KNeighborsClassifier()
pipe = Pipeline(steps=[('kpca', kpca), ('lda', lda), ('knn', knn)])
param_grid = {
    'kpca__n_components': [250, 500, 750],
    'kpca__kernel': ['rbf'],
#    'kpca__kernel': ['poly'],
    'kpca__gamma': [1e-2, 1e-3, 1e-4],
#    'kpca__degree': [2,3,4],
    'knn__n_neighbors': [3, 9],
    'knn__weights': ['uniform'],
    'knn__p': [2]
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1)

folds = KFold(n_splits = 3)
grid = GridSearchCV(pipe, 
                    param_grid=param_grid,
                    scoring= 'f1_macro',
                    cv = folds,
                    verbose = 1,
                    n_jobs=2,
                    return_train_score=True,
                    refit=True)

grid.fit(X2, y)


# printing the optimal accuracy score and hyperparameters
best_score = grid.best_score_
best_hyperparams = grid.best_params_
cv_results = grid.cv_results_
df_cv_results = pd.DataFrame(cv_results)
df_cv_results.to_csv("results_rbf3.csv", header=True)