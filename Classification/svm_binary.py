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


df = pd.read_csv('dataset_L7_binary.csv')
df.replace('?',-99999, inplace=True)
#df.drop(['random'], 1, inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

clf = svm.SVC(kernel = 'linear')

clf.fit(X_train, y_train)
svm_predictions = clf.predict(X_test)

print('BINARY CLASSIFICATION L7')
print('')
  
# model accuracy for X_test   
accuracy = clf.score(X_test, y_test)
print("Accuracy =", accuracy)
print('')
#cm = confusion_matrix(y_test, svm_predictions)
#print("confusion matrix")
#print(cm)
#print('')

report = metrics.classification_report(y_test,clf.predict(X_test))
print("Report")
print(report)
print()

su_vec = clf.support_vectors_
print('support vectors')
print(su_vec)
print('')
su_vec_n = clf.n_support_
print('Number of support vectors for each class')
print(su_vec_n)
print('')
print('total Number of support vectors')
print(sum(su_vec_n))
print('')
weights = clf.coef_
print('The weights associated for each feature')
print(weights)
