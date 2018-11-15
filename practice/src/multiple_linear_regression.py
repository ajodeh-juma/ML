#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
from numpy.linalg import inv, lstsq
from numpy import dot, transpose
from sklearn.linear_model import LinearRegression

# generate train and test data sets
columns=['constant', 'Diameter(in inches)', 'Number of toppings', 'Price(in dollars)']
train=pd.DataFrame([[1, 6, 2, 7],[1, 8, 1, 9], [1, 10, 0, 13], [1, 14, 2, 17.5], [1, 18, 0, 18]], columns=columns)
test=pd.DataFrame([[1, 8, 2, 11],[1, 9, 0, 8.5], [1, 11, 2, 15], [1, 16, 2, 18], [1, 12, 0, 11]], columns=columns)

# find the value of beta which minimizes the cost function



X = train[train.columns[:-1]].as_matrix()
y = train[train.columns[-1]].as_matrix()

print (y)
print(X)

print (dot(inv(dot(transpose(X), X)), dot(transpose(X), y)))

# using NumPy's least squares to solve for beta

print(lstsq(X, y)[0])


# train model

X_train=train[train.columns[1:3]].as_matrix()
y_train=train[train.columns[-1]].as_matrix()

model=LinearRegression()
model.fit(X_train, y_train)

X_test=test[test.columns[1:3]].as_matrix()
y_test=test[test.columns[-1]].as_matrix()

# predict

predictions=model.predict(X_test)
for i, prediction in enumerate(predictions):
	print ("Predicted: %s, Target: %s" % (prediction, y_test[i]))

print ("R-squared: %.2f" % model.score(X_test, y_test))