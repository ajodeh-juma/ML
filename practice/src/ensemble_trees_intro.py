#! /usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from IPython.display import Image
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.cross_validation import train_test_split

# import data
# use seed

SEED = 222
np.random.seed(SEED)

df = pd.read_csv('../data/input.csv')


#TRAINING AND TEST SET

def get_train_test(test_size=0.95):
    """split data into train and test sets"""
    y = 1 * (df.cand_pty_affiliation == 'REP')
    X = df.drop(['cand_pty_affiliation'], axis=1)
    X = pd.get_dummies(X, sparse=True)
    X.drop(X.columns[X.std() == 0], axis=1, inplace=True)
    return train_test_split(X, y, test_size=test_size, random_state=SEED)




#df.cand_pty_affiliation.value_counts(normalize=True).plot(kind='bar', title='share of no. donations')
#plt.show()


def print_graph(clf, feature_names):
    """print decision tree"""
    graph = export_graphviz(
    	clf, 
    	label='root',
    	proportion=True,
    	impurity=False,
    	out_file=None,
    	feature_names=feature_names,
    	class_names={0:'D', 1: 'R'},
    	filled=True,
    	rounded=True
    	)
    graph = pydotplus.graph_from_dot_data(graph)


    
    return Image(graph.create_png())


if __name__ == '__main__':
    xtrain, xtest, ytrain, ytest = get_train_test()
    t1 = DecisionTreeClassifier(max_depth=1, random_state=SEED)
    t1.fit(xtrain, ytrain)
    p = t1.predict_proba(xtest)[:,1]

    print ('Decision tree ROC-AUC score: %.3%' % roc_auc_score(ytest,p))
    print_graph(t1, xtrain.columns)