from sklearn.model_selection import GridSearchCV

def getBestGridSearch(clf,params,Xtrain,ytrain):
  CV = GridSearchCV(clf, params, cv = 4, n_jobs=4) 
  CV.fit(Xtrain, ytrain)
  
  params = CV.best_params_

  print("CV Score: ",CV.best_score_)
  print("Classificator: ",clf)
  print("Params: ",CV.best_params_)
  
  f = open("bestParams_{}.csv".format(clf.__str__().replace("()","")),"w")
  f.write(str(CV.best_params_))
  f.close()
  
  return params 