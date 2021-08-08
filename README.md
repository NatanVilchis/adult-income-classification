# Adult data set income classification

Python 3 project for classifying the adult data set using Random Forest and a Decision Tree classifier.

The goal of this project is to train a sample from the original adult dataset with a classifier model (using *csv/training.csv*) 
and predict in the best way logs from *csv/testing.csv*.


## Required modules
````
sklearn
pandas
category-encoders
imblearn
joblib
````

In case of missing one, do the following for each missing, for example:
````python
pip install sklearn
````
## Run example 
To run the whole process simply run 
````python
python randomForest.py
````
or 
````python
python decisionTree.py
````

To test new models you can reuse the randomForest.py template or decisionTree.py


## Final results
### Random Forest
Accuracy (cross-validation:
0.91
Accuraccy XtrainS:  0.93
Accuraccy XtestS:  0.90

### Decision Tree
Accuracy (cross-validation:
0.87
Accuraccy XtrainS:  0.91
Accuraccy XtestS:  0.89

