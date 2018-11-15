#!/usr/bin/env python

# Author: John Juma <ajodeh.juma@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
import os
from os import path
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.tree import export_graphviz
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
import pydotplus
from IPython.display import Image
import pydot
import matplotlib.pyplot as plt
import argparse
import operator



# for reproducibility
SEED = 222
np.random.seed(SEED)


def random_forest_golubdata(microarray_data):
    """Select features (genes) and classify the GOLUB et al., data for classification of ALL and AML"""



    #### Data preparation ####
    # read the file as a dataframe
    gb = pd.read_csv(microarray_data, sep="\t")

    #rdf =  gb.T
    #rdf = gb.drop('Gene Description', axis=1)
    #rdf = rdf.set_index('Gene Accession Number').T

    #rownames = list(rdf.index)

    # dataframe for experiments (having gene expression values)
    #expr_list = [x for x in rownames if 'call' not in x]
    #expr_df = rdf.loc[expr_list]

    # dataframe for calls/labels
    #lbs_df = pd.DataFrame()
    #lbs = [l for l in rownames if 'call' in l]
    #lbs_df = rdf.loc[lbs]
    #lbs_df = lbs_df.apply(lambda x: x.astype(str).str.upper())



    # append the classes dataframe to the experimental dataframe
    #g = lbs_df.apply(pd.Series.value_counts, axis=1)[['A', 'M', 'P']].fillna(0)
    #class_list = pd.Series(list(g.idxmax(axis=1)))
    #expr_df['class'] = class_list.values

    #df = expr_df


    # get list of experimental conditions
    column_list = gb.columns
    samples_list = [s for s in column_list if 'call' not in s]
    labels_list = [s for s in column_list if 'call' in s]

    # drop the columns with expression values and get max counts for classes into a dataframe

    df_labels = pd.DataFrame()
    labels = gb.drop(samples_list, axis=1)
    g = labels.apply(pd.Series.value_counts, axis=1)[['A', 'M', 'P']].fillna(0)
    df_labels['Gene Accession Number'] = gb['Gene Accession Number']
    df_labels['class'] = g.idxmax(axis=1)



    # drop all the columns having calls and merge dataframes on Gene Accession Number
    samples = gb.drop(labels_list, axis=1)
    df = pd.merge(samples, df_labels, on='Gene Accession Number')

    # Labels are the values we want to predict using one-hot encoding/binaries
    labels = preprocessing.LabelBinarizer()
    labels = labels.fit_transform(list(df['class']))

    # check for missing values
    print (df.isnull().values.any())
    T = df.T
    rnames = list(T.index)
    the_df = T.loc[[l for l in rnames if 'class' not in l]]
    l_df = T.loc[[l for l in rnames if 'class' in l]]

    the_labels = list(T.loc['class'])
    nn_labels = preprocessing.LabelBinarizer()
    nnlbs = l_df.values.tolist()
    print (nnlbs)


    print (list(df['class']))

    n_labels = list(T.loc['class'].T)
    n_labels = preprocessing.LabelBinarizer().fit_transform(n_labels)


    LBS = preprocessing.LabelBinarizer().fit_transform(the_labels)

    n_features = the_df.iloc[2:]
    n_features_list = list(n_features.columns)
    n_features = np.array(n_features)

    n_features_l2_norm = preprocessing.normalize(n_features, norm='l2')


    #nX_train, nX_test, ny_train, ny_test = train_test_split(n_features_l2_norm, labels, test_size=0.25, random_state=SEED)



    # remove the class label column
    df = df.drop(['Gene Accession Number', 'Gene Description', 'class'], axis=1)

    # get column names
    feature_list = list(df.columns)

    #### Data preprocessing ####
    # convert to np array
    features = np.array(df)

    #print (n_features.shape, LBS.shape, n_labels.shape, labels.shape, features.shape)


    # normalize the data

    '''objective of normalization is to enhance the similarity of genes sharing a common
    expression pattern throughout the data but in different ranges of absolute expression values'''

    # using log2 transformation
    features_l2_norm = preprocessing.normalize(features, norm='l2')


    # using RobustScaler (IQR)
    '''scaling using IQR normalization procedure: removes the median and scales the data according
    to the quantile range'''

    features_iqr_norm = RobustScaler(quantile_range=(25, 75)).fit_transform(features)

    # using standard scaler
    features_std_norm = StandardScaler().fit_transform(features)

    # split data into test and training datasets
    X_train, X_test, y_train, y_test = train_test_split(features_l2_norm, labels, test_size=0.25, random_state=SEED)


    # instantiate model with 50 decision trees
    rf = RandomForestClassifier(n_estimators=50, random_state=SEED, max_depth=3)

    # train the model on training data
    rf.fit(X_train, y_train)


    # use the forest's predict method on the test data
    predictions = rf.predict(X_test)

    # calculate the absolute errors
    errors = abs(predictions - y_test)

    # print out the mean absolute error (MAE)
    print ('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # calculate mean absolute percentage error (MAPE)
    #mape=100*(errors/y_test)

    # calculate and display accuracy
    #accuracy = 100-np.mean(mape)
    #print ('Accuracy:', round(accuracy, 2), '%.')


    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file='tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('microarray-tree.png')

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
    important_indices = [feature_list.index('14'), feature_list.index('23'), feature_list.index('5')]

    train_important = X_train[:, important_indices]
    test_important = X_test[:, important_indices]

    # train the random forest
    rf_most_important.fit(train_important, y_train)

    # make predictions and determine error
    predictions = rf_most_important.predict(test_important)
    errors = abs(predictions - y_test)

    # display the performance metrics
    print ('Mean Absolute Error:', round(np.mean(errors),2), 'degrees.')

    #mape = np.mean(100*(errors/y_test))

    #print ('Accuracy:', round(accuracy, 2), '%.')


    plt.style.use('fivethirtyeight')

    # list all locations to plot
    x_values = list(range(len(importances)))

    # make a bar chart
    plt.bar(x_values, importances, orientation='vertical')

    # tick labels
    plt.xticks(x_values, feature_list, rotation='vertical')

    plt.ylabel('Importance'); plt.xlabel('Variable')
    plt.title('Variable Importance')
    plt.show()


parser=argparse.ArgumentParser()
helpstr = """python golubALL_AML.py [options]"""
parser.add_argument('-m', '--microarray',    help="csv file having microarray data")
args=parser.parse_args()

if args.microarray != None:
    inputfilename = args.microarray
else:
    sys.stderr.write("Please specify input file (microarray data)!\n")
    sys.exit(2)


res = random_forest_golubdata(inputfilename)
