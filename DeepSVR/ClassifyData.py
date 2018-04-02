import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from keras.models import model_from_json
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


class ClassifyData:
    """Input processed data to classify variants as: Somatic, Ambigious, or Fail 


    """

    def __init__(self, solid_tumor, sample_file_path, header):
        """Assemble pandas.Dataframe of data

            Args:
                solid_tumor (bool): True if solid tumor, False otherwise (i.e. hematologic tumor).
        """
        
        self.classify_samples(solid_tumor, sample_file_path, header)
    
    
    def classify_samples(self, solid_tumor, sample_file_path, header):
        """classify processed data using classifier

            Args:
                classifier (str): trained classifier located in the folder
           
                samples_file_path (str): File path of tab-separated
                             file outlining the tumor bam path,
                             normal bam path, and manual review
                             sites file path (this should be a
                             one-based tsv file containing
                             chromosome, start, and stop),
                             disease, reference fasta file path
                header (bool): True if header False otherwise.
        """
        # Pull in model from output folder
        json_file = open('data/deep_learning_classifier.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights("data/model.h5")
        print("Loaded model from disk")
        print()
    
        processed_data = pd.read_pickle('Output/train.pkl')
        processed_data = processed_data[processed_data.columns.drop(list(processed_data.filter(regex='disease')))]
        
        if solid_tumor:
            processed_data['solid_tumor'] = 1
        else:
            processed_data['solid_tumor'] = 0
            
        X = processed_data.sort_index(axis=1).astype(float).values
        
        calls = pd.read_pickle('Output/call.pkl')
        Y = pd.get_dummies(calls).astype(float).values
        
        probabilities = loaded_model.predict_proba(X)
        
        probs_df = pd.DataFrame(probabilities, columns=['Ambiguous', 'Fail', 'Somatic'], index=processed_data.index)
        probs_df['Max'] = probs_df[['Ambiguous', 'Fail', 'Somatic']].max(axis=1)
        probs_df['Call'] = pd.np.where(probs_df["Max"] == probs_df["Ambiguous"], "A",
                   pd.np.where(probs_df["Max"] == probs_df["Somatic"], "S",
                   pd.np.where(probs_df["Max"] == probs_df["Fail"], "F", 'NONE')))

        probs_df.to_csv("Output/predictions.tsv", sep='\t', header=True)
