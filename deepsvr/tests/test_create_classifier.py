from unittest import TestCase
from deepsvr.CreateClassifier import create_classifier
import os


class TestCreate_classifier(TestCase):
    def test_create_classifier(self):
        train_data_path = './deepsvr/tests/test_data/training_data/train.pkl'
        label_path = './deepsvr/tests/test_data/training_data/call.pkl'
        model_out_path = './deepsvr/tests/test_data/training_data/model.json'
        weights_out_path = './deepsvr/tests/test_data/training_data/' \
                           'model_weights.hd5'
        create_classifier(train_data_path, label_path, model_out_path,
                          weights_out_path)
        self.assertTrue(os.path.exists(model_out_path))
        self.assertTrue(os.path.exists(weights_out_path))
