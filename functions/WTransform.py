import pandas as pd
import numpy as np 
 
def changeWorkclass(row):
  gov = ['Local-gov', 'State-gov','Federal-gov']
  selfEmp = ['Self-emp-inc','Self-emp-not-inc' ]
  private = ['Private']
  other = ['Without-pay']
  if row in private:
    return "Private"
  if row in gov:
    return "Gov"
  if row in selfEmp:
    return "selfEmp"
  if row in other:
    return "Other"
    
def changeMaritalStatus(row):
  married = ["Married-civ-spouse","Married-spouse-absent","Married-AF-spouse"]
  separated = ["Divorced","Separated",] 
  single = ["Never-married"]
  special = ["Widowed",]
  if row in married:
    return "Married"
  if row in separated:
    return "Separated"
  elif row in single:
    return "Single"
  elif row in special:
    return "Special"
    

 

def changeEducation(row):
  HS = ['HS-grad','11th','10th','9th','12th',]
  elementary = ['7th-8th', '1st-4th',    '5th-6th',]
  if row in HS:
    return "HS"
  if row in elementary:
    return "Elementary"
  else: 
    return row   

def changeRelationship(row):
  married = ['Wife','Husband',]
  notMarried = [ 'Own-child', 'Not-in-family',   'Unmarried',] 
  other = ['Other-relative']
  if row in married:
    return "Married"
  if row in notMarried:
    return "Not Married"
  if row in other:
    return "Other"

def changeNativeCountry(row):
  EUA = ["United-States"]
  if row in EUA:
    return "United-States"
  else:
    return "Other country"

def changeRace(row):
  white = ["White"]
  black = ["Black"]
  if row in white:
    return "White"
  if row in black:
    return "Black"
  else:
    return "Other"

def changeOccupation(row):
  prof = ['Prof-specialty', ]
  other = ['Armed-Forces', ]
  serv = ['Priv-house-serv','Protective-serv','Other-service', ]
  if row in prof:
    return "Prof"
  if row in other:
    return "Other"
  if row in serv:
    return "Serv"
  else:
    return row 
  
def WTransform(X):
  W = pd.DataFrame(X)


  W["maritalStatus"] = W["maritalStatus"].apply(changeMaritalStatus)
  W["workclass"] = W["workclass"].apply(changeWorkclass)
  W["relationship"] = W["relationship"].apply(changeRelationship)
  W["nativeCountry"] = W["nativeCountry"].apply(changeNativeCountry)
  W["race"] = W["race"].apply(changeRace)
  W["occupation"] = W["occupation"].apply(changeOccupation) 
  W["education"] = W["education"].apply(changeEducation) 

  
  W['capitalGain'] = W['capitalGain'].apply(lambda x: np.log((x + 1)))
  W['capitalLoss'] = W['capitalLoss'].apply(lambda x: np.log((x + 1)))
  
  return W