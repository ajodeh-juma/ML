#!/usr/bin/env python

from __future__ import print_function
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load the dataset and read
df = pd.read_csv('../data/SMSSpamCollection', delimiter='\t')

print ('Number of spam messages:', df[df['label'] == 'spam']['label'].count())
print ('Number of ham messages:', df[df['label'] == 'ham']['label'].count())

lb = preprocessing.LabelBinarizer()

# split the data into training and test (75% as training and 25% test)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
y_train = np.array([number[0] for number in lb.fit_transform(y_train)])
# feature extraction and preprocessing

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# create an instance of LogisticRegression and train the model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
#scores = cross_val_score(classifier, X_train, y_train, cv=5)
#print (np.mean(scores), scores)
#predictions = classifier.predict(X_test)

precisions =cross_val_score(classifier, X_train, y_train, cv=5, scoring="precision")
print (np.mean(precisions), precisions)

recalls =cross_val_score(classifier, X_train, y_train, cv=5, scoring="recall")
print (np.mean(recalls), recalls)

#print (predictions[:5])
#for i, prediction in enumerate(predictions[:10]):

#	print ('Prediction: %s Message: %s' % (prediction, X_test_raw[i]))




#y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
#y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]

#confusion_matrix =  confusion_matrix(y_test, y_pred)
#print (confusion_matrix)
#plt.matshow(confusion_matrix)
#plt.title('Confusion matrix')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()
