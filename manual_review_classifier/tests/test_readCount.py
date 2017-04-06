from unittest import TestCase

from manual_review_classifier.ReadCount import ReadCount

import pickle


class TestReadCount(TestCase):
    def test__parse(self):
        valid_dict = pickle.load(open(
            'manual_review_classifier/tests/tst1/tst1_normal_valid_dict.pkl',
            'rb'))
        test_dict = ReadCount('manual_review_classifier/tests/tst1/' + \
                              'tst1_normal.counts').read_count_dict
        self.assertEqual(valid_dict, test_dict)
