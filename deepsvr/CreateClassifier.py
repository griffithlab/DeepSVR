# Import Tools
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_classifier(training_file_path, label_file_path, model_out_file_path,
                      weights_out_file_path):
    """Create a deep learning classifier to perform somatic variant refinement
    on automated variant calls

        Args:
            training_file_path (str): File path of training data
                produced by the deepsvr prepare_data command.
            label_file_path (str): File path of label file
                produced by the deepsvr prepare_data command.
            model_out_file_path (str): File path to save the model object json.
            weights_out_file_path (str): File path to save the model weights
                hd5 file.
    """
    # Create Data
    training_data = pd.read_pickle(training_file_path)

    # Get Labels
    Y = pd.read_pickle(label_file_path).replace('g', 'f')
    Y.sort_index(inplace=True)
    print('One-hot encoding labels.')
    Y = pd.get_dummies(Y).astype(float).values
    # Get training data as numpy array, remove reviews
    # because of non overlap
    X = training_data.sort_index(axis=1).astype(float).values

    print('Labels one-hot encoded shape is: ', Y.shape)
    print('Training shape is: ', X.shape)

    # Create Model
    model = Sequential()
    model.add(Dense(59, input_dim=59, kernel_initializer='normal',
                    activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(3, kernel_initializer='normal', activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # Fit Model
    model.fit(X, Y, epochs=1000, batch_size=2000, verbose=1)

    # Serialize Model to JSON
    json_model = model.to_json()
    with open(model_out_file_path, 'w') as json_file:
        json_file.write(json_model)

    # Serialize weights to HDF5
    model.save_weights(weights_out_file_path)
    print("Saved model to disk")
    print("Model path: ", model_out_file_path)
    print("Model weights path ", weights_out_file_path)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Succesfully trained model!')
    return model
