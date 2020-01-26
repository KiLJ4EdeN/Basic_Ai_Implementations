import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold
from sklearn.metrics import confusion_matrix
import scipy.io
import pickle
import pandas as pd
import numpy.matlib
from sklearn.pipeline import Pipeline
start_time = time.time()
# loading thr dataset
data = pd.read_csv('mushroom_csv.csv')
le = LabelEncoder()
# encoding the class column
data['class'] = le.fit_transform(data['class'])
# removing class from feature vector
Y = data['class'].values.reshape(-1, 1)
data = data.drop('class', 1)
#print(data.head(5))
# encoding string features into binary ones
encoded_data = pd.get_dummies(data)
#print(encoded_data.head(5))
X = np.array(encoded_data.iloc[:, :])
# shape of feature vector
print(X.shape)
# shape of class
Y = np.ravel(Y)
print(Y.shape)
# featured are encoded so there is no need to scale.
# scaler = sklearn.preprocessing.StandardScaler(copy = True, with_mean = True, with_std = True, )
# mikowski with p=2 equals euclidean dist.
knn = KNeighborsClassifier(n_neighbors = 1, weights = 'uniform', algorithm = 'auto', leaf_size = 30, p = 2, metric = 'minkowski', metric_params = None, n_jobs = None, )
cv = StratifiedKFold(n_splits = 10, shuffle=True, random_state=42)
# performing 10 fold cross validation
scores = cross_val_score(knn, X, Y , cv = cv)
ypred = cross_val_predict(knn, X, Y , cv = cv)
# computing confusion
cm = confusion_matrix(Y, ypred)
print(cm)
print('Accurracy: %0.4f (+/- %0.4f)' % (scores.mean(),scores.std()*2))
print('---%s seconds---' % (time.time()-start_time))