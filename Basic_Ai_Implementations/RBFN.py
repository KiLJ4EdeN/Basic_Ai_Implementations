import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('mushroom_csv.csv')
le = LabelEncoder()
data['class'] = le.fit_transform(data['class'])
Y = data['class'].values.reshape(-1, 1)
data = data.drop('class', 1)
encoded_data = pd.get_dummies(data)
X = np.array(encoded_data.iloc[:, :])
print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True, stratify=Y)
print('Poisonous Mushroom Count:', sum(Y==1))
print('Edible Mushroom Count:', sum(Y==0))
print("================================")
print('Class 1 test_size:', sum(Y_test==1))
print('Class 2 test_size:', sum(Y_test==0))	
K_cent= 48
# finding k clusters in the data
km= KMeans(n_clusters= K_cent, max_iter= 100)
km.fit(X_train)
cent= km.cluster_centers_
max=0 
# finding max distance for sigma calculation
for i in range(K_cent):
	for j in range(K_cent):
		# sqrt(sum(x.^2))
		d= np.linalg.norm(cent[i]-cent[j])
		if(d> max):
			max= d
d= max
# sigma calculation
sigma= d/math.sqrt(2*K_cent)
shape= X_train.shape
row= shape[0]
column= K_cent
G= np.empty((row,column), dtype= float)
# finding distances between train matrix and centroids from k means
for i in range(row):
	for j in range(column):
		dist= np.linalg.norm(X_train[i]-cent[j])
		G[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))
# computing weights
GTG= np.dot(G.T,G)
GTG_inv= np.linalg.inv(GTG)
fac= np.dot(GTG_inv,G.T)
W= np.dot(fac,Y_train)
row= X_test.shape[0]
column= K_cent
G_test= np.empty((row,column), dtype= float)
for i in range(row):
	for j in range(column):
		dist= np.linalg.norm(X_test[i]-cent[j])
		G_test[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))
prediction= np.dot(G_test,W)
prediction= 0.5*(np.sign(prediction-0.5)+1)

score= accuracy_score(prediction,Y_test)
cm = confusion_matrix(Y_test, prediction)
print(cm)
acc = ((cm[0][0]+cm[1][1])/np.sum(cm))*100
specificity = (cm[0][0]/(cm[0][0]+cm[0][1]))*100
sensitivity = (cm[1][1]/(cm[1][1]+cm[1][0]))*100
print('Cluster Size:', K_cent)
print('Test Accuracy: %.2f' % (acc))
print('Sensitivity : %.2f' % (sensitivity))
print('Specificity : %.2f' % (specificity))