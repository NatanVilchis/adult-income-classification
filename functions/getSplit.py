from sklearn.model_selection import train_test_split

def getSplit(X,y,test_size):
  XtrainS, XtestS, ytrainS, ytestS = train_test_split( X, y, test_size = test_size)
  return XtrainS, XtestS, ytrainS, ytestS
