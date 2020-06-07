import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import cv2 
import os  
from random import shuffle 
from tqdm import tqdm
import pandas as pd
from sklearn.svm import SVC 

df = pd.read_csv('dataset_L7_bin_size_8.csv')
df.replace('?',-99999, inplace=True)
#df.drop(['random'], 1, inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.transform(X_test)

svm_model_linear = SVC(kernel = 'linear',C=1).fit(X_train_minmax, y_train) 
svm_predictions = svm_model_linear.predict(X_test_minmax)

print('MULTI-CLASS CLASSIFICATION_L7 BIN SIZE 8')
print('')
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test_minmax, y_test)
print("Accuracy =", accuracy)
print('')

report = metrics.classification_report(y_test,svm_model_linear.predict(X_test_minmax))
print("Report")
print(report)
print()
# creating a confusion matrix 
#cm = confusion_matrix(y_test, svm_predictions)
#print("confusion matrix")
#print(cm)
print('')
su_vec = svm_model_linear.support_vectors_
print('support vectors')
print(su_vec)
print('')
su_vec_n = svm_model_linear.n_support_
print('Number of support vectors for each class')
print(su_vec_n)
print('')
print('total Number of support vectors')
print(sum(su_vec_n))
print('')
weights = svm_model_linear.coef_
print('The weights associated for each feature')
print(weights)

