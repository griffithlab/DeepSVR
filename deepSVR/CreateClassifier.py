#3.10.2018
#!/usr/bin/env python3
"""Script that builds the classifier for cli code

Usage: python3 CreateClassifier.py <training data path>

Input:
 1) training data path = path to pkl dataframe
     
     NOTE: Run Prepare data on training data to get formatted input .pkl

Output:
 1) classifier stored as .json() file

Note:

"""



# Import Tools
import numpy as np
import pandas as pd
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from keras.models import model_from_json
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
import json
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from manual_review_classifier.ClassifierPlots import create_reliability_diagram, create_roc_curve, create_feature_importance_plot, make_model_output_plot
from manual_review_classifier.Analysis import determine_feature_importance, print_accuracy_and_classification_report, predict_classes, get_somatic_error_type, calculate_kappa


import click
@click.command()
@click.help_option('-h', '--help')
@click.option('--training-file-path', '-tfp', default=None,
              help='Specify file to be used to train a new classifier.')


def main(training_file_path):
    # Create Data
    training_data = pd.read_pickle(training_file_path)
    training_data = training_data.replace('g','f')

    # Get Labels
    Y = pd.get_dummies(training_data.call).astype(float).values
    # Get training data as numpy array, remove reviews because of non overlap
    X = training_data.sort_index(axis=1).drop('call', axis=1).astype(float).values

    print('Y Shape is: ', Y.shape)
    print('X Shape is: ', X.shape)

    # Set Model Seed
    seed = 7
    np.random.seed(seed)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    # Create Model
    model = Sequential()
    model.add(Dense(59, input_dim=59, kernel_initializer='normal', activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dense(3, kernel_initializer='normal', activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit Model
    model.fit(X, Y, epochs=1000, batch_size=2000, verbose=0)

    # Serialize Model to JSON
    json_model = model.to_json()
    with open('data/deep_learning_classifier.json', 'w') as json_file:
        json_file.write(json_model)

    # Serialize weights to HDF5
    model.save_weights("data/model.h5")
    print("Saved model to disk")

if __name__ == '__main__':

    main()
