import os
import pandas as pd
import numpy as np

from zero_one_based_conversion import convert
from manual_review_classifier.ReadCount import ReadCount

from sklearn import preprocessing


class PrepareData:
    """Prepare data for classification or training from bam and manual review
        files


    """

    def __init__(self, samples_file_path, header, output_dir_path,
                 skip_readcount):
        """Assemble pandas.Dataframe of data

            Args:
                samples_file_path (str): File path of tab-separated
                                         file outlining the tumor bam path,
                                         normal bam path, and manual review
                                         sites file path (this should be a
                                         one-based tsv file containing
                                         chromosome, start, and stop),
                                         disease, reference fasta file path
                header (bool): True if header False otherwise.
                output_dir_path (str): path for output directory
                skip_readcount (bool): skip the read counting step by reading
                                       in the read count files from a prior run
                                       in the output directory.
        """
        self._parse_samples_file(samples_file_path, header)
        self.output_dir_path = output_dir_path
        self.training_data = pd.DataFrame()
        self.categorical_columns = list()
        self._run_bam_readcount(skip_readcount)

    def _parse_samples_file(self, samples_file_path, header):
        """Parse samples

            Args:
                samples_file_path (str): File path of tab-separated
                                         file outlining the tumor bam path,
                                         normal bam path, and manual review
                                         sites file path (this should be a
                                         one-based tsv file containing
                                         chromosome, start, and stop),
                                         disease, reference fasta file path
                header (bool): True if header False otherwise.
        """
        with open(samples_file_path) as f:
            samples = f.readlines()
            samples = [x.strip() for x in samples]
            samples = [x.split('\t') for x in samples]
            if header:
                samples.pop(0)
        self.samples = samples

    def _run_bam_readcount(self, skip_readcount):
        """Run bam-readcount on created sites file. Concatenate review calls.

            Args:
                skip_readcount (bool): Skip the read counting step by reading
                                       in the read count files from a prior run
                                       in the output directory.
        """
        output_dir_path = os.path.join(self.output_dir_path, 'readcounts')
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        self.review = pd.DataFrame(columns=['chromosome', 'start', 'stop',
                                            'ref', 'var', 'call', 'disease'])
        for sample in self.samples:
            print('-----------------------------------------------------'
                  '\nStarting on sample {0}\n'.format(sample[0]))
            reviewer_specified_in_sample = False
            sites_file_path = os.path.join(output_dir_path, sample[0] + '.sites')
            review = self._parse_review_file(sample[3], sites_file_path, sample[0])
            reviewer_in_bed_file = False
            review['disease'] = sample[4]
            self.review = pd.concat([self.review, review], ignore_index=True)
            bed_one_based_f_path = sample[3] + '.one_based'
            print('Processing tumor bam file:\n\t{0}'.format(sample[1]))
            tumor_readcount_file_path = '{0}/{1}_tumor' \
                                        '.readcounts'.format(output_dir_path,
                                                             sample[0])
            if not skip_readcount:
                os.system('bam-readcount -i -w 0 -l {0} -f {1} '
                          '{2} > {3}'.format(sites_file_path, sample[5],
                                             sample[1],
                                             tumor_readcount_file_path))

            tumor_rc = ReadCount(tumor_readcount_file_path)

            tumor_data = tumor_rc.compute_variant_metrics(bed_one_based_f_path,
                                                          'tumor',
                                                          reviewer_in_bed_file,
                                                          sample[4])

            print('Processing normal bam file:\n\t{0}'.format(sample[2]))
            normal_readcount_file_path = '{0}/{1}_normal' \
                                         '.readcounts'.format(output_dir_path,
                                                              sample[0])
            if not skip_readcount:
                os.system('bam-readcount -i -w 0 -l {0} -f {1} '
                          '{2} > {3}'.format(sites_file_path, sample[5],
                                             sample[2],
                                             normal_readcount_file_path))
            normal_rc = ReadCount(normal_readcount_file_path)
            normal_data = normal_rc.\
                compute_variant_metrics(bed_one_based_f_path, 'normal',
                                        reviewer_in_bed_file, sample[4])
            if len(tumor_data) != len(normal_data):
                raise ValueError(
                    'Dataframes cannot be merged. They are differing lengths.')
            
            individual_df = pd.merge(tumor_data, normal_data,
                                         on=['chromosome', 'start', 'stop',
                                             'ref', 'var', 'call', 'disease'])

            individual_df.index = (sample[0] + '~' + individual_df.chromosome +
                                   ':' + individual_df.start.map(str) + '-' +
                                   individual_df.stop.map(str) +
                                   individual_df.ref + '>' +
                                   individual_df['var'])
            self.training_data = pd.concat([self.training_data, individual_df])
        self.training_data.drop(['chromosome', 'start', 'stop', 'ref', 'var'],
                                axis=1, inplace=True)
        self._perform_one_hot_encoding('disease')
        self.calls = self.training_data.pop('call')

        # normalize continuous variables
        columns = ['normal_VAF', 'normal_depth', 'normal_other_bases_count',
                   'normal_ref_avg_basequality',
                   'normal_ref_avg_clipped_length',
                   'normal_ref_avg_distance_to_effective_3p_end',
                   'normal_ref_avg_distance_to_q2_start_in_q2_reads',
                   'normal_ref_avg_mapping_quality',
                   'normal_ref_avg_num_mismaches_as_fraction',
                   'normal_ref_avg_pos_as_fraction',
                   'normal_ref_avg_se_mapping_quality',
                   'normal_ref_avg_sum_mismatch_qualities', 'normal_ref_count',
                   'normal_ref_num_minus_strand', 'normal_ref_num_plus_strand',
                   'normal_ref_num_q2_containing_reads',
                   'normal_var_avg_basequality',
                   'normal_var_avg_clipped_length',
                   'normal_var_avg_distance_to_effective_3p_end',
                   'normal_var_avg_distance_to_q2_start_in_q2_reads',
                   'normal_var_avg_mapping_quality',
                   'normal_var_avg_num_mismaches_as_fraction',
                   'normal_var_avg_pos_as_fraction',
                   'normal_var_avg_se_mapping_quality',
                   'normal_var_avg_sum_mismatch_qualities', 'normal_var_count',
                   'normal_var_num_minus_strand', 'normal_var_num_plus_strand',
                   'normal_var_num_q2_containing_reads', 'tumor_VAF',
                   'tumor_depth',
                   'tumor_other_bases_count', 'tumor_ref_avg_basequality',
                   'tumor_ref_avg_clipped_length',
                   'tumor_ref_avg_distance_to_effective_3p_end',
                   'tumor_ref_avg_distance_to_q2_start_in_q2_reads',
                   'tumor_ref_avg_mapping_quality',
                   'tumor_ref_avg_num_mismaches_as_fraction',
                   'tumor_ref_avg_pos_as_fraction',
                   'tumor_ref_avg_se_mapping_quality',
                   'tumor_ref_avg_sum_mismatch_qualities', 'tumor_ref_count',
                   'tumor_ref_num_minus_strand', 'tumor_ref_num_plus_strand',
                   'tumor_ref_num_q2_containing_reads',
                   'tumor_var_avg_basequality',
                   'tumor_var_avg_clipped_length',
                   'tumor_var_avg_distance_to_effective_3p_end',
                   'tumor_var_avg_distance_to_q2_start_in_q2_reads',
                   'tumor_var_avg_mapping_quality',
                   'tumor_var_avg_num_mismaches_as_fraction',
                   'tumor_var_avg_pos_as_fraction',
                   'tumor_var_avg_se_mapping_quality',
                   'tumor_var_avg_sum_mismatch_qualities', 'tumor_var_count',
                   'tumor_var_num_minus_strand', 'tumor_var_num_plus_strand',
                   'tumor_var_num_q2_containing_reads']
        to_normalize = self.training_data[columns]
        
        # Source http://stackoverflow.com/a/26415620
        x = to_normalize.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        scaled = pd.DataFrame(x_scaled, index=self.training_data.index,
                              columns=columns)
        self.training_data = pd.concat(
            [self.training_data[self.categorical_columns], scaled], axis=1)
        self.training_data.to_pickle(
            os.path.join(self.output_dir_path, 'train.pkl'))
        self.calls.to_pickle(os.path.join(self.output_dir_path, 'call.pkl'))

    def _perform_one_hot_encoding(self, column):
        """perform one-hot encoding on categorical variables

            Args:
                column (str): Column name to perform encoding on

        """
        get_dummies = pd.get_dummies(
            self.training_data[column], prefix=column)
        self.categorical_columns += get_dummies.columns.values.tolist()
        self.training_data = pd.concat([self.training_data, get_dummies],
                                       axis=1)
        self.training_data.drop(column, axis=1, inplace=True)


    def _parse_review_file(self, manual_review_file_path, sites_file_path,
                           sample_name):
        manual_review = pd.read_csv(manual_review_file_path, sep='\t')
        manual_review.columns = map(str.lower, manual_review.columns)
        manual_review.rename(columns={'reference': 'ref', 'variant': 'var'},
                             inplace=True)
        manual_review = manual_review[['chromosome', 'start', 'stop',
                                           'ref', 'var', 'call']]
        manual_review = manual_review.apply(self._convert_one_based, axis=1)
        manual_review = manual_review.replace('', np.nan).dropna(how='all')
        manual_review[['chromosome', 'start', 'stop']].to_csv(sites_file_path,
                                                              sep='\t',
                                                              index=False,
                                                              header=False)
        manual_review.to_csv(manual_review_file_path+'.one_based', sep='\t',
                             index=False, header=True)
        return manual_review

    def _convert_one_based(self, row):
        return convert.coordinate_system('\t'.join(map(str, row.values)),
                                         'to_one_based').strip().split('\t')

