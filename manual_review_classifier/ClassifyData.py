import os
import pandas as pd
import numpy as np
import sklearn


from sklearn import preprocessing


class ClassifyData:
    """Input processed data to classify variants into: Somatic, Ambigious, and Fail 


    """

    def __init__(self, processed_data, classifier, hematologic_tumor):
        """Assemble pandas.Dataframe of data

            Args:
                processed_data (str): File path of tab-separated
                                         file outlining the tumor bam path,
                                         normal bam path, and manual review
                                         sites file path (this should be a
                                         one-based tsv file containing
                                         chromosome, start, and stop),
                                         disease, reference fasta file path
                classifier (str): trained classifier located in the folder
                solid_tumor (bool): True if solid tumor, False otherwise (i.e. hematologic tumor).
        """
        self._parse_samples_file(samples_file_path, header)
        self.out_dir_path = out_dir_path
        self.training_data = pd.DataFrame()
        self.categorical_columns = list()
        self._run_bam_readcount(skip_readcount)

    def parse_samples(processed_data):
        """parse processed data to return samples

            Args:
                processed_data (str): File path of tab-separated
                                         file outlining the tumor bam path,
                                         normal bam path, and manual review
                                         sites file path (this should be a
                                         one-based tsv file containing
                                         chromosome, start, and stop),
                                         disease, reference fasta file path
        """
        
        with open(processed_data) as f:
            samples = f.readlines()
            samples = [x.strip() for x in samples]
            samples = [x.split('\t') for x in samples]
        self.samples = samples
        
    def classify_samples(classifier):
        """classify processed data using classifier

            Args:
                classifier (str): trained classifier located in the folder
        """
        
        model = model_from_json(open('deep_learning_classifier.json').read())
        
        if solid_tumor:
            processed_data['solid_tumor'] = 1
        else:
            processed_data['solid_tumor'] = 0
        
        X = samples.sort_index(axis=1).astype(float).values
        
        return estimator.predict_proba(X)
        
        
        
        
        
        
        
        