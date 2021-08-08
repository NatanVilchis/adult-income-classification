from sklearn.neighbors import LocalOutlierFactor
from functions.getTrain import getTrain
from functions.savePredictionCSV import savePredictionCSV
from functions.sampler import sampler
from functions.getSplit import getSplit
from functions.saveBestModel import saveBestModel
from functions.printAccTrainTestAll import printAccTrainTestAll
from joblib import dump 
  

def fullProcess(fileModel,saveFilenameCSV,params,clf,filenameTrain="csv/training.csv",filenameTest="csv/testing.csv"):
  columns, X, y = getTrain(filenameTrain)

  X, y = sampler(X,y) 

  Outlier = LocalOutlierFactor(metric="chebyshev",n_neighbors=2)
  yHat = Outlier.fit_predict(X)
  mask = yHat != -1
  X, y = X[mask, :], y[mask] 
  XtrainS, XtestS, ytrainS, ytestS, = getSplit(X,y, test_size = 0.20)
  
  dump(XtrainS,"functions/Xtrain.joblib")
  dump(XtestS,"functions/Xtest.joblib")
  dump(ytrainS,"functions/ytrain.joblib")
  dump(ytestS,"functions/ytest.joblib")
  saveBestModel(fileModel, clf, params, XtrainS, ytrainS)
  printAccTrainTestAll(fileModel, XtrainS, XtestS, ytrainS, ytestS, X, y)
  savePredictionCSV(filenameTest, saveFilenameCSV, fileModel, columns) 