from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import imblearn

class Helper:
  def __init__(self):
    print("Helper")
    pass
  
  def plotConfusionMatrix(self, test_y, y_pred, title="Confusion Matrix", norm=True):
      plt.figure()
      confusionMatrix = metrics.confusion_matrix(test_y, y_pred)
      print(confusionMatrix)
      if norm:
        confusionMatrix = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]      
      print(confusionMatrix)
      sns.heatmap(confusionMatrix, annot=True, cbar=False, fmt = '.2f', cmap = 'Oranges')
      plt.title(title)
      plt.xlabel('True output' )
      plt.ylabel('Predicted output')    
      plt.show()
      
  
  def printResults(self, test_y, y_pred):
      print("recall_score: '{0}'" .format(metrics.recall_score(test_y, y_pred)))
      print("precision_score: '{0}'" .format(metrics.precision_score(test_y, y_pred)))
      print("accuracy_score: '{0}'" .format(metrics.accuracy_score(test_y, y_pred)))
      print("f1_score: '{0}'" .format(metrics.f1_score(test_y, y_pred)))
      # print(metrics.confusion_matrix(test_y, y_pred), "\n")
      
      accuracy = metrics.accuracy_score(test_y, y_pred)
      recall = metrics.recall_score(test_y, y_pred)
      precision = metrics.precision_score(test_y, y_pred)
      f1 = metrics.f1_score(test_y, y_pred)
      return accuracy, recall, precision, f1
  
  def printResults2(self, y_test, y_pred):
      accuracy = metrics.accuracy_score(y_test,y_pred)
      precision = metrics.precision_score(y_test,y_pred)
      recall = metrics.recall_score(y_test,y_pred)
      f1 = metrics.f1_score(y_test,y_pred)
      geometric_mean = imblearn.metrics.geometric_mean_score(y_test,y_pred)
      balanced_accuracy = metrics.balanced_accuracy_score(y_test,y_pred) 
       
      print("Recall: "+str(round((recall*100),2))+"%")
      print("Precision: "+str(round((precision*100),2))+"%")
      print("Accuracy: "+str(round((accuracy*100),2))+"%")
      print("F1: "+str(round((f1*100),2))+"%")
      print("Geometric Mean: "+str(round((geometric_mean*100),2))+"%")
      print("Balanced Accuracy: "+str(round((balanced_accuracy*100),2))+"%")
      
