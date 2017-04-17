from unittest import TestCase

from manual_review_classifier.ReadCount import ReadCount

import pandas as pd
import pickle

TEST_DATA_ROOT = 'manual_review_classifier/tests/tst1/'

class TestReadCount(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.read_count = ReadCount( TEST_DATA_ROOT + 'tst1_normal.counts')

    def test__parse(self):
        valid_dict = pickle.load(open(TEST_DATA_ROOT +
                                      'tst1_normal_valid_dict.pkl',
                                      'rb'))
        test_dict = self.read_count.read_count_dict
        self.assertEqual(valid_dict, test_dict)

    def test_compute_variant_metrics(self):
        data = self.read_count.compute_variant_metrics(TEST_DATA_ROOT +
                                                       'tst1_full.bed', 'normal_1')
        # print(data.columns)
        self.assertTrue(all(data.columns ==
                            ['chromosome', 'ref', 'normal_1_ref_count',
                             'normal_1_ref_avg_mapping_quality',
                             'normal_1_ref_avg_basequality',
                             'normal_1_ref_avg_se_mapping_quality',
                             'normal_1_ref_num_plus_strand', 'normal_1_ref_num_minus_strand',
                             'normal_1_ref_avg_pos_as_fraction',
                             'normal_1_ref_avg_num_mismaches_as_fraction',
                             'normal_1_ref_avg_sum_mismatch_qualities',
                             'normal_1_ref_num_q2_containing_reads',
                             'normal_1_ref_avg_distance_to_q2_start_in_q2_reads',
                             'normal_1_ref_avg_clipped_length',
                             'normal_1_ref_avg_distance_to_effective_3p_end',
                             'normal_1_var_count',
                             'normal_1_var_avg_mapping_quality', 'normal_1_var_avg_basequality',
                             'normal_1_var_avg_se_mapping_quality',
                             'normal_1_var_num_plus_strand',
                             'normal_1_var_num_minus_strand', 'normal_1_var_avg_pos_as_fraction',
                             'normal_1_var_avg_num_mismaches_as_fraction',
                             'normal_1_var_avg_sum_mismatch_qualities',
                             'normal_1_var_num_q2_containing_reads',
                             'normal_1_var_avg_distance_to_q2_start_in_q2_reads',
                             'normal_1_var_avg_clipped_length',
                             'normal_1_var_avg_distance_to_effective_3p_end',
                             'normal_1_other_bases_count', 'var',
                             'stop', 'start', 'normal_1_depth']))
# ['chromosome', 'start', 'stop',
# 'ref', 'var', 'depth','tumor_vaf',
# 'tumor_var_base_count',
# 'tumor_var_avg_mapping_quality'
# 'tumor_var_avg_basequality',
# 'tumor_var_avg_se_mapping_quality',
# 'tumor_var_num_plus_strand',
# 'tumor_var_num_minus_strand',
# 'tumor_var_avg_pos_as_fraction',
# 'tumor_var_avg_num_mismaches_as_fraction',
# 'tumor_var_avg_sum_mismatch_qualities',
# 'tumor_var_num_q2_containing_reads',
# 'tumor_var_avg_distance_to_q2_start_in_q2_reads',
# 'tumor_var_avg_clipped_length',
# 'tumor_var_avg_distance_to_effective_3p_end',
# 'tumor_ref_base_count',
# 'tumor_ref_avg_mapping_quality'
# 'tumor_ref_avg_basequality',
# 'tumor_ref_avg_se_mapping_quality',
# 'tumor_ref_num_plus_strand',
# 'tumor_ref_num_minus_strand',
# 'tumor_ref_avg_pos_as_fraction',
# 'tumor_ref_avg_num_mismaches_as_fraction',
# 'tumor_ref_avg_sum_mismatch_qualities',
# 'tumor_ref_num_q2_containing_reads',
# 'tumor_ref_avg_distance_to_q2_start_in_q2_reads',
# 'tumor_ref_avg_clipped_length',
# 'tumor_ref_avg_distance_to_effective_3p_end',
# 'tumor_other_bases_count',
# 'normal_vaf',
# 'normal_var_base_count',
# 'normal_var_avg_mapping_quality'
# 'normal_var_avg_basequality',
# 'normal_var_avg_se_mapping_quality',
# 'normal_var_num_plus_strand',
# 'normal_var_num_minus_strand',
# 'normal_var_avg_pos_as_fraction',
# 'normal_var_avg_num_mismaches_as_fraction',
# 'normal_var_avg_sum_mismatch_qualities',
# 'normal_var_num_q2_containing_reads',
# 'normal_var_avg_distance_to_q2_start_in_q2_reads',
# 'normal_var_avg_clipped_length',
# 'normal_var_avg_distance_to_effective_3p_end',
# 'normal_ref_base_count',
# 'normal_ref_avg_mapping_quality'
# 'normal_ref_avg_basequality',
# 'normal_ref_avg_se_mapping_quality',
# 'normal_ref_num_plus_strand',
# 'normal_ref_num_minus_strand',
# 'normal_ref_avg_pos_as_fraction',
# 'normal_ref_avg_num_mismaches_as_fraction',
# 'normal_ref_avg_sum_mismatch_qualities',
# 'normal_ref_num_q2_containing_reads',
# 'normal_ref_avg_distance_to_q2_start_in_q2_reads',
# 'normal_ref_avg_clipped_length',
# 'normal_ref_avg_distance_to_effective_3p_end',
# 'normal_other_bases_count']
