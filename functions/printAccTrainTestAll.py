from joblib import load 
from sklearn import metrics 

def printAccTrainTestAll(fileModel, XtrainS, XtestS, ytrainS, ytestS, X, y):
  clf = load(fileModel)
  ypredict = clf.predict( XtrainS )
  accuracyTrain = metrics.accuracy_score(ytrainS, ypredict )
  print("Accuraccy XtrainS: ",accuracyTrain)
  
  ypredict = clf.predict( XtestS )
  accuracyTest = metrics.accuracy_score( ytestS, ypredict )
  print("Accuraccy XtestS: ", accuracyTest)
  
  ypredict = clf.predict( X )
  accuracyAll = metrics.accuracy_score( y, ypredict )
  print("Accuraccy X: ", accuracyAll) 