import pandas as pd
from joblib import load 
from functions.WTransform import WTransform

def toLabel(row):
  if row == 0: 
    row = "<=50K"
  else: 
    row = ">50K"  
  return row 

def savePredictionCSV(filenameTest, saveFilename, fileModel,columns,preprocesorFilename="functions/pipelineTrain.joblib"):
  column_names = [
                'age', 
                'workclass', 
                'fnlwgt', 
                'education', 
                'educationNum',
                'maritalStatus',
                'occupation', 
                'relationship', 
                'race', 
                'sex',
                'capitalGain', 
                'capitalLoss', 
                'hoursPerWeek', 
                'nativeCountry',
                'id'
                ]
  
  DB = pd.read_csv(filenameTest,names=column_names,header=0)
  X = WTransform(DB[columns])
  
  
  pipeTransform = load(preprocesorFilename)
  X = pipeTransform.transform(X)
  
  clf = load(fileModel)
  
  
  y = clf.predict(X)
  newDB = pd.DataFrame({
    "id" : [i+1 for i in range(len(y))],
    "income" : y
  })
  newDB["income"] = newDB["income"].apply(toLabel) 
  newDB.to_csv(saveFilename,header=True,index=False)