#!/usr/bin/env python

from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
import pydot
import matplotlib.pyplot as plt
import datetime

# for reproducibility
SEED = 222
np.random.seed(SEED)

features = pd.read_csv('../data/temps.csv')

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Labels are the values we want to predict
labels = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# convert to np array
features = np.array(features)


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=SEED)


# The baseline predictions are the historical averages
baseline_preds = X_test[:, feature_list.index('average')]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - y_test)

print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=10, random_state=SEED, max_depth=3)

# train the model on training data
rf.fit(X_train, y_train)

# use the forest's predict method on the test data
predictions = rf.predict(X_test)

# calculate the absolute errors
errors = abs(predictions - y_test)

# print out the mean absolute error (MAE)
print ('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# calculate mean absolute percentage error (MAPE)
mape=100*(errors/y_test)

# calculate and display accuracy
accuracy = 100-np.mean(mape)
print ('Accuracy:', round(accuracy, 2), '%.')


# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file='tree2.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree2.dot')
# Write graph to a png file
graph.write_png('tree1.png')

# get numerical feature importances
importances = list(rf.feature_importances_)

# list of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# sort the feature importances by the most important first

feature_importances = sorted(feature_importances, key=lambda x:x[1], reverse=True)

# print out feature and importance

[print ('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# new random forest with only two most important variables

rf_most_important = RandomForestRegressor(n_estimators=1000, random_state=42)

# extract the two most important features

important_indices = [feature_list.index('temp_1'), feature_list.index('average')]

train_important = X_train[:, important_indices]
test_important = X_test[:, important_indices]

# train the random forest
rf_most_important.fit(train_important, y_train)

# make predictions and determine error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - y_test)

# display the performance metrics
print ('Mean Absolute Error:', round(np.mean(errors),2), 'degrees.')

mape = np.mean(100*(errors/y_test))

print ('Accuracy:', round(accuracy, 2), '%.')

plt.style.use('fivethirtyeight')

# list all locations to plot
x_values = list(range(len(importances)))

# make a bar chart
plt.bar(x_values, importances, orientation='vertical')

# tick labels
plt.xticks(x_values, feature_list, rotation='vertical')

plt.ylabel('Importance'); plt.xlabel('Variable')
plt.title('Variable Importance')

# dates of training vales
months = features[:, features_list.index('month')]
days = features[:, features_list.index('day')]
years = features[:, features_list.index('year')]


# list and then convert to dataframe object
dates = [str(int(year))+'-'+str(int(month))+'-'+str(int(day))]