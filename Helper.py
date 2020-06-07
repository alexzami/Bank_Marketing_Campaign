from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Helper:
  def __init__(self):
    print("Helper")
    pass
  
  def plotConfusionMatrix(self, test_y, y_pred, title="Confusion Matrix"):
      plt.figure()
      confusionMatrix = metrics.confusion_matrix(test_y, y_pred)
      print(confusionMatrix)
      cm_norm = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]
      print(cm_norm)
      sns.heatmap(cm_norm, annot=True, cbar=False, fmt = '.2f', cmap = 'Oranges')
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