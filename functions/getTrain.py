from pandas import read_csv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump 
from category_encoders.m_estimate import MEstimateEncoder
from functions.WTransform import WTransform 

def getTrain(filenameCSV,preprocesorFilename="functions/pipelineTrain.joblib",scaler=MinMaxScaler,encoder=MEstimateEncoder,):
  namesCSV= [
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
                'income'
                ]
   
  DB = read_csv(filenameCSV,names=namesCSV,header=0)  
  #DB  = DB[DB["maritalStatus"] == "Married-civ-spouse"]
  #DB  = DB[DB["sex"] == "Male"]
  #DB  = DB[(DB["age"] >= 0) & (DB["age"] < 24) ]
  #DB  = DB[(DB["age"] >= 24) & (DB["age"] < 26) ]
   
  numericFeatures = [
              'age', 
              'fnlwgt', 
              #'educationNum', 
              'capitalGain', 
              'capitalLoss',
              'hoursPerWeek'
              ]
  
  categoricalFeatures = [
    'workclass', 
    'education', 
    'maritalStatus', 
    'occupation', 
    'relationship',
    'race', 
    'sex', 
    'nativeCountry',
  ]
   

  columns = numericFeatures + categoricalFeatures
  X = WTransform(DB[columns])
  
  
  transformerCategorical = Pipeline(
    steps=[
      ('encoder', encoder())
    ])

  transformerNumerical = Pipeline(
    steps=[
      ('scaler', scaler()),

    ])

  preprocessorPipe = ColumnTransformer(
    transformers=[
      ('num', transformerNumerical, numericFeatures),
      ('cat', transformerCategorical, categoricalFeatures),
      ])


  y = DB['income']
  y = LabelEncoder().fit_transform(y)
  
  preprocessor = preprocessorPipe.fit(X,y)
  X = preprocessor.transform(X)
  
  dump(preprocessor,preprocesorFilename)
  
  return columns, X, y