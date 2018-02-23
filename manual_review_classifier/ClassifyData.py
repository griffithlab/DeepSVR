import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import model_from_json


class ClassifyData:
    """Input processed data to classify variants into: Somatic, Ambigious, and Fail 


    """

    def __init__(self, solid_tumor, classifier, samples_file_path, header, output_dir_path):
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
                solid_tumor (bool): True if solid tumor, False otherwise (i.e. hematologic tumor).
                classifier (str): path for classifier (default is 'deep_learning_classifier.json')
                out_dir_path (str): path for output
                
        """
        
        self.output_dir_path = output_dir_path
        self._parse_samples_file(samples_file_path, header)
        self._classify_samples(solid_tumor, classifier, output_dir_path)
        
        

    def _parse_samples_file(self, samples_file_path, header):
        """Parse samples

            Args:
                samples_file_path (str): File path of tab-separated
                                         file outlining the tumor bam path,
                                         normal bam path, and manual review
                                         sites file path (this should be a
                                         one-based tsv file containing
                                         chromosome, start, and stop),
                                         disease, reference fasta file path
                header (bool): True if header False otherwise.
        """
        with open(samples_file_path) as f:
            samples = f.readlines()
            samples = [x.strip() for x in samples]
            samples = [x.split('\t') for x in samples]
            if header:
                samples.pop(0)
        self.samples = samples

    
    def _classify_samples(self, solid_tumor, classifier, output_dir_path):
        """classify processed data using classifier

            Args:
                solid_tumor (bool): True if solid tumor, False otherwise (i.e. hematologic tumor).
                classifier (str): path for classifier (default is 'deep_learning_classifier.json')
                out_dir_path (str): path for output

        """
        
        estimator = model_from_json(open('deep_learning_classifier.json').read())
        
        processed_data = pd.read_pickle('output/train.pkl')

        if solid_tumor:
            processed_data['solid_tumor'] = 1
        else:
            processed_data['solid_tumor'] = 0
        X = processed_data.sort_index(axis=1).drop(['disease_BRC','reviewer_1'], 
            axis =1).astype(float).values
        
        probabilities = estimator.predict_proba(X)
        
        out_dir_path = output_dir_path
        predictions = pd.DataFrame(probabilities)
        predictions.to_csv(out_dir_path + "predictions.tsv")

        
        
        
        
        
        
        
        