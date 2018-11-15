#!/usr/bin/env python

from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile, os.path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# get current working directory and create a data directory to store files
cwd=os.getcwd()
datadir=os.path.join(os.path.split(cwd)[0], 'data')
if not os.path.exists(datadir):
	os.makedirs(datadir)

# generate train and test data sets
columns=['Diameter(in inches)','Price(in dollars)']
train=pd.DataFrame([[6, 7],[8, 9], [10, 13], [14, 17.5], [18, 18]], columns=columns)
test=pd.DataFrame([[6, 7],[8, 9], [10, 13], [16, 17.5]], columns=columns)

# write data sets as csv files

train_filename=os.path.join(datadir,'trainset.csv')
train.to_csv(train_filename, index=False)

test_filename=os.path.join(datadir,'testset.csv')
test.to_csv(test_filename, index=False)

# train
X_train=train[train.columns[0:1]].as_matrix()
y_train=train[train.columns[-1]].as_matrix()

# test
X_test=test[test.columns[0:1]].as_matrix()
y_test=test[test.columns[-1]].as_matrix()


regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)
#plt.show()

quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)



# model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# plot
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()
#plt.savefig('polynomial_vs_linear.pdf')

print(X_train)
print(X_train_quadratic)
print(X_test)
print(X_test_quadratic)

print("Simple linear regression r-squared: %f" % (regressor.score(X_test, y_test)))
print("Quadratic regression r-squared: %f" % (regressor_quadratic.score(X_test_quadratic, y_test)))