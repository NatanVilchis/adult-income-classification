from sklearn.ensemble import RandomForestClassifier
from functions.fullProcess import fullProcess 


fileModel = "randomForest.joblib"
saveFilenameCSV = "randomForestPrediction.csv"
 
n_estimators = [i for i in range(1,250,10)]
max_depth = [i for i in range(1,250,10)]
criterion= ["entropy"]
min_samples_leaf = [5]
params = {
  'n_estimators' : n_estimators,
  'max_depth' : max_depth,
  "min_samples_leaf":min_samples_leaf,
  "criterion":criterion 
} 
clf = RandomForestClassifier()

fullProcess(fileModel,saveFilenameCSV,params,clf)