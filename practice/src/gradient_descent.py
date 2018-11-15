#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score

"""using stochastic gradient to estimate the parameters of a model using scikit-learn SGDRegressor"""

# load dataset, split the data into training and testing sets
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)



#X_train = np.array(X_train).reshape((-1, 1))
#y_train = np.array(y_train).reshape((-1, 1))

#X_test = np.array(X_test).reshape((-1, 1))
#y_test = np.array(y_test).reshape((-1, 1))

# scale the features 
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
y_train = np.array(y_train).reshape((-1, 1))
y_train = y_scaler.fit_transform(y_train)


X_test = X_scaler.transform(X_test)
y_test = np.array(y_test).reshape((-1, 1))
y_test = y_scaler.transform(y_test)





# train the estimator, evaluate using cross validation and test

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print ('Cross validation r-squared score:', scores)
print ('Average cross validation r-squared score:', np.mean(scores))

regressor.fit(X_train, y_train)
print ('Test set r-squared score', regressor.score(X_test, y_test))
