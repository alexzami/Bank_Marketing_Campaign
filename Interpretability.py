from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source
from sklearn.tree import export_graphviz
from IPython.display import SVG
from IPython.display import display                               
from ipywidgets import interactive
from sklearn.metrics import accuracy_score

import os
os.environ["PATH"] += os.pathsep + "C:\\Users\\user_a\\Anaconda3\\pkgs\\graphviz-2.38-hfd603c8_2\\Library\\bin"
import graphviz

class Interpretability:
    def __init__(self):
        print("Interpretability")
        pass


    """
    Logistic Regression
    """
    def plotFeaturesCoefficientGlobal(self, model, feature_names):
        """
        plot the weights (coefficient of the features = feature importance)
        Global
        """
        weights = model.coef_
        model_weights = pd.DataFrame({ 'features': list(feature_names),'weights': list(weights[0])})
        model_weights = model_weights.reindex(model_weights['weights'].abs().sort_values(ascending=False).index) #Sort by absolute value
        model_weights = model_weights[(model_weights["weights"] != 0)]    
        print("Number of features:",len(model_weights.values))

        plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        sns.barplot(x="weights", y="features", data=model_weights)
        plt.title("Intercept (Bias): "+str(model.intercept_[0]),loc='right')
        plt.xticks(rotation=90)
        plt.show()
    
    def plotFeaturesCoefficientLocal(self, model, x_test, y_test, y_pred, feature_names, instance=0):
        random_instance = x_test[instance]
        predicted_proba_test = model.predict_proba(x_test)
        print("Original Class:", str(y_test[instance]) + ", Predicted Class:", str(y_pred[instance]), "with probability of", str(predicted_proba_test[instance][y_pred[instance]]))
        title = "Original Class:" + str(y_test[instance]) + ", Predicted Class:" + str(y_pred[instance]) + "with probability of" + str(predicted_proba_test[instance][y_pred[instance]])
        weights = model.coef_
        summation = sum(weights[0]*random_instance)
        bias = model.intercept_[0]
        res = ""
        if (summation + bias > 0):
            res = " > 0 -> 1"
        else:
            res = " <= 0 -> 0"
        print("Sum(weights*instance): "+str(summation)+" + Intercept (Bias): "+str(bias)+" = "+ str(summation+bias)+ res)
        # title = title + "\n" + "Sum(weights*instance): "+str(summation)+" + Intercept (Bias): "+str(bias)+" = "+ str(summation+bias)+ res
        model_weights = pd.DataFrame({ 'features': list(feature_names),'weights*values': list(weights[0]*random_instance)})
        model_weights = model_weights.reindex(model_weights['weights*values'].abs().sort_values(ascending=False).index) #Sort by absolute value
        model_weights = model_weights[(model_weights["weights*values"] != 0)]    
        print("Number of features:",len(model_weights.values))
        # title = title + "\n" + "Number of features:",len(model_weights.values)
        plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        sns.barplot(x="weights*values", y="features", data=model_weights)
        plt.title(title)
        plt.xticks(rotation=90)
        plt.show()

    """
    Decision trees
    """
    def plot_DecisionTree(self, depth, x_train, y_train, x_test, y_test, feature_names, plotNamePath="./results/BankMarket"):
        estimator = DecisionTreeClassifier(random_state = 0 , criterion = 'gini', max_depth = depth)
        estimator.fit(x_train, y_train)
        dot_data = export_graphviz(estimator, out_file=None, feature_names=feature_names, class_names=['yes','no'], filled = True)
        graph = graphviz.Source(dot_data) 
        graph.render(plotNamePath)
        print(accuracy_score(y_test, estimator.predict(x_test)))
        return estimator

    def plot_DecisionTree_feature_importance(self, depth, x_train, y_train, feature_names):
        estimator = DecisionTreeClassifier(random_state = 0, criterion = 'gini', max_depth = depth)
        estimator.fit(x_train, y_train)
        weights = estimator.feature_importances_
        model_weights = pd.DataFrame({ 'features': list(feature_names),'weights': list(weights)})
        model_weights = model_weights.sort_values(by='weights', ascending=False)
        plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
        sns.barplot(x="weights", y="features", data=model_weights)
        plt.xticks(rotation=90)
        plt.show()
        return estimator

