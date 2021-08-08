from imblearn.over_sampling import RandomOverSampler, SMOTE 

def sampler(X,y):
  S = SMOTE()
  S.fit(X, y)
  XSampled, ySampled = S.fit_resample(X, y)  
  
  return XSampled, ySampled