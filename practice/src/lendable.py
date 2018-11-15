#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import pandas as pd
import tempfile, os.path
from mlxtend.preprocessing import standardize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/home/juma/Downloads/Diamond_data_set.csv', sep = ',', index_col=False)

print(df.describe())

for grp, cert in df.groupby('certification'):
	print (grp, cert.median())

X = df.carat.values
y = df.price.values
X_train = X[0:77]
y_train = y[0:77]
X_train = np.array(X).reshape((-1, 1))
y_train = np.array(y).reshape((-1, 1))

plt.plot(X, y, 'k.')
plt.grid(True)
#plt.show()

X_test = X[78:]
y_test = y[78:]

X_test = np.array(X).reshape((-1, 1))
y_test = np.array(y).reshape((-1, 1))
print ("\nBuilding the model\n")
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

for i, prediction in enumerate(predictions):
	print ('Predicted: %s, Target: %s' % (prediction, y_test[i]))
print("\n")
print("R-squared: %.2f" % model.score(X_test, y_test))



columns=['text']
column2=['name']
df1=pd.DataFrame(['alice'], columns=columns)
df2=pd.DataFrame()
df2['text']=df1['text']

df2.columns=['phrase']

df2['price']=[10000]
df3=pd.DataFrame()

df3=df2[df2['price']<7000]




columns=['text', 'subjectivity', 'polarity', 'restaurant']

