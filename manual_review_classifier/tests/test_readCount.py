from unittest import TestCase
from manual_review_classifier.ReadCount import ReadCount

TEST_DATA_ROOT = 'manual_review_classifier/tests/test_data/'


class TestReadCount(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.read_count = ReadCount(TEST_DATA_ROOT + 'training_data/readcounts'
                                                    '/tst1_normal.readcounts')
        cls.t_read_count = ReadCount(
            TEST_DATA_ROOT + 'training_data/readcounts'
                             '/tst1_tumor.readcounts')

    def test__parse(self):
        self.assertEqual(443, len(self.read_count.read_count_dict))

    def test_compute_variant_metrics(self):
        data = self.read_count.compute_variant_metrics(TEST_DATA_ROOT +
                                                       'tst1.review.one_based',
                                                       'normal_1', False,
                                                       'BRC')
        self.assertEqual(36, len(data.columns))
        t_data = self.t_read_count.compute_variant_metrics(TEST_DATA_ROOT +
                                                           'tst1.review.'
                                                           'one_based',
                                                           'tumor', True,
                                                           'BRC')
        self.assertEqual(37, len(t_data.columns))
        # Check that the read counts add up to the depth
        rcount = len(data[data.normal_1_depth ==
                          data.normal_1_other_bases_count +
                          data.normal_1_ref_count + data.normal_1_var_count])
        self.assertTrue(rcount == len(data))
        self.assertFalse(any(data.normal_1_VAF.isnull()))
