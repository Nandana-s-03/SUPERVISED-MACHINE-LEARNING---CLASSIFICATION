import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,accuracy_score
iris=load_iris()
x=iris.data
print(x)
y=iris.target
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
log_regr=LogisticRegression()
log_regr.fit(x_train,y_train)
y_pred=log_regr.predict(x_test)
print(y_pred)
precision_score_log=precision_score(y_test,y_pred,average='macro')
print(precision_score_log)
