import os
import pandas as pd
from keras.models import model_from_json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def classify_samples(prepared_data_path, model_file_path, model_weights_path,
                     predictions_out_path):
    """classify processed data using classifier

        Args:
            prepared_data_path (str): Specify the 'train.pkl' file produced by
                                      the 'prepare_data' to perform inference
                                      on. Ignore the call.pkl used in training
                                      classifiers.
            model_file_path (str): Specify the file path for the model json
                                   file. Created by the train_classifier
                                   command.
            model_weights_path (str): Specify the file path for the model
                                      weights file. Created by the
                                      train_classifier command.
            predictions_out_path (str): Specify the file path for the
                                        predictions tab separated file.
    """
    # Pull in model from output folder
    json_file = open(model_file_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(model_weights_path)
    print("Loaded model from disk")
    print()

    processed_data = pd.read_pickle(prepared_data_path)
    X = processed_data.sort_index(axis=1).astype(float).values
    probabilities = loaded_model.predict_proba(X)
    probs_df = pd.DataFrame(probabilities,
                            columns=['Ambiguous', 'Fail', 'Somatic'],
                            index=processed_data.index)
    probs_df['Max'] = probs_df[['Ambiguous',
                                'Fail',
                                'Somatic']].max(axis=1)
    probs_df['Call'] = pd.np.where(
        probs_df["Max"] == probs_df["Ambiguous"], "A",
        pd.np.where(probs_df["Max"] == probs_df["Somatic"], "S",
                    pd.np.where(probs_df["Max"] == probs_df["Fail"], "F",
                                'NONE')
                    )
    )

    probs_df.to_csv(predictions_out_path, sep='\t', header=True)
