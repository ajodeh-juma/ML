#!/usr/bin/env python

from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile, os.path
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split, cross_val_score

df=pd.read_csv("../data/winequality-red.csv", sep=';')

# summary statistics for each column(variable) in the dataframe
print (df.describe())


# explore the data: plot scatter plots between explanatory variable(s) and response variable
plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()

# deciding which explanatory variables to include in the model: use DataFrame.corr()
# calculates a pairwise correlation matrix

print(df.corr)

#The correlation matrix confirms that the
#strongest positive correlation is between the alcohol and quality, and that quality
#is negatively correlated with volatile acidity, an attribute that can cause wine to
#taste like vinegar. To summarize, we have hypothesized that good wines have high
#alcohol content and do not taste like vinegar.

#-------- split the data into training and testing sets -------#

X = df[list(df.columns)[:-1]]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)

#-------- train the regressor ---------------------------------#
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#-------- evaluate its predictions ----------------------------#
y_predictions = regressor.predict(X_test)
print("R-squared: ", regressor.score(X_test, y_test))

#35 percent of the variance in the test set is explained by the model

#use cross-validation to produce a better estimate of the estimator's performance
# each cross-validation round/iteration trains and tests different partitions of the data to reduce variablity
regressor = LinearRegression()
scores = cross_val_score(regressor, X, y, cv=5)
print(scores.mean(), scores)
regressor.fit(X_train, y_train)
predictions=regressor.predict(X_test)

#for i, score in enumerate(predictions):
#	print (i, score)