import numpy as np
import pandas as pd
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
import json


import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from manual_review_classifier.ClassifierPlots import create_reliability_diagram, create_roc_curve, create_feature_importance_plot, make_model_output_plot
from manual_review_classifier.Analysis import determine_feature_importance, print_accuracy_and_classification_report, predict_classes, get_somatic_error_type, calculate_kappa

class CreateClassifier:
    """Create classifier from training data


    """

    def __init__(self, samples_file_path, header, out_dir_path,
                 skip_readcount):
        """Assemble pandas.Dataframe of data

            Args:
                samples_file_path (str): File path of tab-separated
                                         file outlining the tumor bam path,
                                         normal bam path, and manual review
                                         sites file path (this should be a
                                         one-based tsv file containing
                                         chromosome, start, and stop),
                                         disease, reference fasta file path
                header (bool): True if header False otherwise.
                out_dir_path (str): path for output directory
                skip_readcount (bool): skip the read counting step by reading
                                       in the read count files from a prior run
                                       in the output directory.
        """
        self._create_classifier(self, training_data)


    def _create_classifier(self, training_data):
        """Parse samples

            Args:
                samples_file_path (str): File path of tab-separated
                                         file outlining the tumor bam path,
                                         normal bam path, and manual review
                                         sites file path (this should be a
                                         one-based tsv file containing
                                         chromosome, start, and stop),
                                         disease, reference fasta file path
                training_data (str): string to data used for training model
        """
        
        
        sns.set_style("white")
        sns.set_context('talk')

        training_data = pd.read_pickle('training_data')

        s_v_b = training_data.replace('g','f')
        s_v_b['solid_tumor'] = s_v_b[['disease_GST', 'disease_MPNST', 'disease_SCLC',
                                      'disease_breast', 'disease_colorectal', 
                                      'disease_glioblastoma', 'disease_melanoma']].apply(any, axis=1).astype(int)
        s_v_b.drop(['disease_AML', 'disease_GST', 'disease_MPNST', 'disease_SCLC',
                    'disease_breast', 'disease_colorectal', 'disease_glioblastoma',
                    'disease_lymphoma', 'disease_melanoma'], axis=1, inplace=True)

        # Get Labels
        Y = pd.get_dummies(three_class.call).astype(float).values
        # Get training data as numpy array, remove reviews because of non overlap
        X = s_v_b.sort_index(axis=1).drop(['call', 'reviewer_1',
                                           'reviewer_2', 'reviewer_3', 
                                           'reviewer_4'], axis=1).astype(float).values

        # define baseline model
        def model():
            # create model
            model = Sequential()
            model.add(Dense(59, input_dim=59, kernel_initializer='normal', activation='tanh', kernel_regularizer=l2(0.001)))
            model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.001)))
            model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.001)))
            model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.001)))
            model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.001)))
            model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        estimator = KerasClassifier(build_fn=model, epochs=1000, batch_size=2000, verbose=0)

        estimator.fit(X, Y)

        json_model = estimator.model.to_json()
        open('deep_learning_classifier.json', 'w').write(json_model)

        from keras.models import model_from_json

        model = model_from_json(open('deep_learning_classifier.json').read())
        
        return model
