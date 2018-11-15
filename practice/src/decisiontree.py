#! /usr/bin/env python

#from __future__ import print_function
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('../data/ad-dataset/ad.data', header=None)
explanatory_variable_columns = set(df.columns.values)
response_variable_column = df[len(df.columns.values)-1]

# the last column describes the targets
explanatory_variable_columns.remove(len(df.columns.values)-1)

y = [1 if e == 'ad.' else 0 for e in response_variable_column]
X = df[list(explanatory_variable_columns)]
# Replace missing values with negative values (-1)
X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# create a pipeline and an instance of DecisionTreeClassifier using entropy as criteion to build information gain heuristic
#steps = [('clf', DecisionTreeClassifier(criterion='entropy'))]
#pipeline = Pipeline(steps)


#specify the hyperparameter space for the grid search
#parameters = {
#	'clf__max_depth': (150, 155, 160),
#	'clf__min_samples_split': (1, 2, 3),
#	'clf__min_samples_leaf': (1, 2, 3)
#}

# using Rnadom Forest Classifier
steps = [('clf', RandomForestClassifier(criterion='entropy'))]
pipeline = Pipeline(steps)

parameters = {
	'clf__n_estimators': (5, 10, 20, 50),
	'clf__max_depth': (50, 150, 250),
	'clf__min_samples_split': (1, 2, 3),
	'clf__min_samples_leaf': (1, 2, 3)
}




# set GridSearchCV to maximize the model's F1 score
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
grid_search.fit(X_train, y_train)
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
	print '\t%s: %r' % (param_name, best_parameters[param_name])

predictions = grid_search.predict(X_test)
print classification_report(y_test, predictions)










