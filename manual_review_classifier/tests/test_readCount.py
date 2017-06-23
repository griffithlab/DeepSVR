from unittest import TestCase

from manual_review_classifier.ReadCount import ReadCount

# import pandas as pd
import pickle

TEST_DATA_ROOT = 'test_data/'

class TestReadCount(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.read_count = ReadCount( TEST_DATA_ROOT + 'training_data/readcounts'
                                                     '/tst1_normal.readcounts')

    def test__parse(self):
        self.assertEqual(443, len(self.read_count.read_count_dict))

    def test_compute_variant_metrics(self):
        data = self.read_count.compute_variant_metrics(TEST_DATA_ROOT +
                                                       'tst1.review', 'normal_1')
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
