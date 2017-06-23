from unittest import TestCase
from manual_review_classifier.PrepareData import PrepareData


def file_len(fname):
    """Source: https://stackoverflow.com/q/845058/3862525"""
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

class TestPrepareData(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.samples_noheader = PrepareData('./test_data/samples.noheader.tsv',
                                           False, 'test_data/training_data')
        cls.samples_header = PrepareData('./test_data/samples.tsv',
                                           True, 'test_data/training_data')

    def test__parse_samples_file(self):
        self.assertTrue(len(self.samples_header.samples) == 1)
        self.assertTrue(len(self.samples_noheader.samples) == 1)

    def test__run_bam_readcount(self):
        self.assertEqual(file_len('./test_data/training_data/readcounts'
                                  '/tst1_normal.readcounts'), 443)
        self.assertEqual(file_len('./test_data/training_data/readcounts'
                                  '/tst1_tumor.readcounts'), 443)
