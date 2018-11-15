#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import tempfile, os.path
from mlxtend.preprocessing import standardize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


X = [6, 8, 10, 14, 18]
y = [7, 9, 13, 17.5, 18]

#plt.figure()
#plt.title('Pizza price plotted against diameter')
#plt.xlabel('Diameter in inches' )
#plt.ylabel('Price in dollars')
#plt.plot(X, y, 'k.')
#plt.axis([0,25,0,25])
#plt.grid(True)
#plt.show()

# create and fit the model

X = np.array(X).reshape((-1, 1))
y = np.array(y).reshape((-1, 1))
print(X)
model = LinearRegression()
print (model)
model.fit(X, y)
model.predict(12)[0]

print('A 12 pizza should cost: $%.2f' % model.predict(12)[0])

# compute the residual sum of squares for the model
print("Residual sum of squares: %.2f" % np.mean((model.predict(X) - y) **2))

# calculate the variance
X = [6, 8, 10, 14, 18]
y = [7, 9, 13, 17.5, 18]

xbar=(6+8+10+14+18)/5
ybar=(7+9+13+17.5+18)/5
variance = ((6-xbar)**2 + (8-xbar)**2 + (10-xbar)**2 +(14-xbar)**2 + (18-xbar)**2)/4
print (variance)

# calculate variance using NumPy var
print(np.var(X, ddof=1))


cov=(((6-xbar)*(7-ybar)+(8-xbar)*(9-ybar)+(10-xbar)*(13-ybar)+(14-xbar)*(17.5-ybar)+(18-xbar)*(18-ybar))/4)
print(cov)
print(np.cov(X, y)[0][1])
beta=cov/variance
print(np.mean(y), np.mean(X))
alpha=np.mean(y)-(np.cov(X,y)[0][1]/np.var(X,ddof=1))*np.mean(X)
print(alpha)

X_test=[8,9,11,16,12]
y_test=[11,8.5,15,18,11]
X_test=np.array(X_test).reshape(-1,1)

y_test = np.array(y_test).reshape((-1, 1))
y_predicted=[]

for i in X_test:
	y_pred=alpha+beta*i
	y_predicted.append(y_pred)
print(y_predicted)

#Evaluating the model

# measure the total sum of the squares

sstot=0
for y in y_test:
	sstot+=(np.mean(y_test)-y)**2

# Find the residual sum of squares (cost function)

X = [6, 8, 10, 14, 18]
y = [7, 9, 13, 17.5, 18]
X = np.array(X).reshape((-1, 1))
y = np.array(y).reshape((-1, 1))

model.fit(X,y)
print ("R-squared: %.4f" % model.score(X_test, y_test))