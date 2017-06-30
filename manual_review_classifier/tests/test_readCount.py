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
                                                       'tst1.review',
                                                       'normal_1', False,
                                                       'BRC')
        normal_columns = ['normal_1_ref_count',
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
                          'var', 'call', 'stop', 'start', 'normal_1_depth',
                          'disease', 'normal_1_VAF']
        self.assertTrue(all(data.columns == normal_columns))
        t_data = self.t_read_count.compute_variant_metrics(TEST_DATA_ROOT +
                                                           'tst1.review',
                                                           'tumor', True,
                                                           'BRC')
        tumor_columns = ['tumor_ref_count',
                         'tumor_ref_avg_mapping_quality',
                         'tumor_ref_avg_basequality',
                         'tumor_ref_avg_se_mapping_quality',
                         'tumor_ref_num_plus_strand',
                         'tumor_ref_num_minus_strand',
                         'tumor_ref_avg_pos_as_fraction',
                         'tumor_ref_avg_num_mismaches_as_fraction',
                         'tumor_ref_avg_sum_mismatch_qualities',
                         'tumor_ref_num_q2_containing_reads',
                         'tumor_ref_avg_distance_to_q2_start_in_q2_reads',
                         'tumor_ref_avg_clipped_length',
                         'tumor_ref_avg_distance_to_effective_3p_end',
                         'tumor_var_count',
                         'tumor_var_avg_mapping_quality',
                         'tumor_var_avg_basequality',
                         'tumor_var_avg_se_mapping_quality',
                         'tumor_var_num_plus_strand',
                         'tumor_var_num_minus_strand',
                         'tumor_var_avg_pos_as_fraction',
                         'tumor_var_avg_num_mismaches_as_fraction',
                         'tumor_var_avg_sum_mismatch_qualities',
                         'tumor_var_num_q2_containing_reads',
                         'tumor_var_avg_distance_to_q2_start_in_q2_reads',
                         'tumor_var_avg_clipped_length',
                         'tumor_var_avg_distance_to_effective_3p_end',
                         'tumor_other_bases_count', 'chromosome', 'ref',
                         'var', 'call',
                         'stop', 'start', 'tumor_depth', 'reviewer',
                         'disease', 'tumor_VAF']
        self.assertTrue(all(t_data.columns == tumor_columns))
        # Check that the read counts add up to the depth
        rcount = len(data[data.normal_1_depth ==
                          data.normal_1_other_bases_count +
                          data.normal_1_ref_count + data.normal_1_var_count])
        self.assertTrue(rcount == len(data))
        self.assertFalse(any(data.normal_1_VAF.isnull()))
