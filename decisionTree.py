from sklearn.ensemble import RandomForestClassifier
from functions.fullProcess import fullProcess 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

fileModel = "new.joblib"
saveFilenameCSV = "newPrediction.csv"
 
max_depth = [i for i in range(1,250,10)]
criterion= ["entropy"]
min_samples_leaf = [i for i in range(1,20,1)]
min_samples_split = [2]
params = {
  "max_depth":max_depth,
  "criterion":criterion,
  "min_samples_leaf":min_samples_leaf,
  "min_samples_split":min_samples_split,
} 
clf = DecisionTreeClassifier()

fullProcess(fileModel,saveFilenameCSV,params,clf)