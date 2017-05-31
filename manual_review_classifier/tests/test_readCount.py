from unittest import TestCase

from manual_review_classifier.ReadCount import ReadCount

# import pandas as pd
import pickle

TEST_DATA_ROOT = 'tst1/'

class TestReadCount(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.read_count = ReadCount( TEST_DATA_ROOT + 'tst1_normal.counts')

    def test__parse(self):

        # The below code is used to generate the blessed test data. The below
        # is intended to insure that parsing does not change unexpectedly.
        #
        # rc = ReadCount(
        #     '../manual_review_classifier/tests/tst1/tst1_normal.counts')
        # df = rc.compute_variant_metrics(
        #     '../manual_review_classifier/tests/tst1/tst1_full.bed', 'normal_1')
        # pickle.dump(d, open(
        #     '../manual_review_classifier/tests/tst1/tst1_normal_valid_dict.pkl',
        #     'wb'))

        valid_dict = pickle.load(open(TEST_DATA_ROOT +
                                      'tst1_normal_valid_dict.pkl',
                                      'rb'))
        test_dict = self.read_count.read_count_dict
        self.assertEqual(valid_dict, test_dict)

    def test_compute_variant_metrics(self):
        data = self.read_count.compute_variant_metrics(TEST_DATA_ROOT +
                                                       'tst1_full.bed', 'normal_1')
        self.assertTrue(all(data.columns ==
                            ['normal_1_ref_count',
                             'normal_1_ref_avg_mapping_quality',
                             'normal_1_ref_avg_basequality',
                             'normal_1_ref_avg_se_mapping_quality',
                             'normal_1_ref_num_plus_strand',
                             'normal_1_ref_num_minus_strand',
                             'normal_1_ref_avg_pos_as_fraction',
                             'normal_1_ref_avg_num_mismaches_as_fraction',
                             'normal_1_ref_avg_sum_mismatch_qualities',
                             'normal_1_ref_num_q2_containing_reads',
                             'normal_1_ref_avg_distance_to_q2_start_in_q2_reads',
                             'normal_1_ref_avg_clipped_length',
                             'normal_1_ref_avg_distance_to_effective_3p_end',
                             'normal_1_var_count',
                             'normal_1_var_avg_mapping_quality',
                             'normal_1_var_avg_basequality',
                             'normal_1_var_avg_se_mapping_quality',
                             'normal_1_var_num_plus_strand',
                             'normal_1_var_num_minus_strand',
                             'normal_1_var_avg_pos_as_fraction',
                             'normal_1_var_avg_num_mismaches_as_fraction',
                             'normal_1_var_avg_sum_mismatch_qualities',
                             'normal_1_var_num_q2_containing_reads',
                             'normal_1_var_avg_distance_to_q2_start_in_q2_reads',
                             'normal_1_var_avg_clipped_length',
                             'normal_1_var_avg_distance_to_effective_3p_end',
                             'normal_1_other_bases_count', 'chromosome', 'ref',
                             'var', 'call',
                             'stop', 'start', 'normal_1_depth', 'normal_1_VAF']))
        # Check that the read counts add up to the depth
        self.assertTrue(len(data[data.normal_1_depth==data.normal_1_other_bases_count+data.normal_1_ref_count+data.normal_1_var_count]) == len(data))
        self.assertFalse(any(data.normal_1_VAF.isnull()))
