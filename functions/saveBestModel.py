from functions.getBestGridSearch import getBestGridSearch
from joblib import dump

def saveBestModel(fileModel, clf, params, XtrainS, ytrainS):
  params = getBestGridSearch(clf, params, XtrainS, ytrainS)

  clf.set_params(**params)
  clf.fit(XtrainS,ytrainS)
  dump(clf, fileModel) 