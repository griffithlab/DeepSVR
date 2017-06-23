import os
import pandas as pd
import numpy as np

from zero_one_based_conversion import convert

class PrepareData:
    """Prepare data for classification or training from bam and manual review
        files


    """

    def __init__(self, samples_file_path, header, out_dir_path):
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
        """
        self._parse_samples_file(samples_file_path, header)
        self.out_dir_path = out_dir_path
        self._run_bam_readcount()

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

    def _run_bam_readcount(self):
        """Run bam-readcount on created sites file. Concatenate review calls.

            Args:
                out_dir_path (str): Path of directory for all data output
        """
        out_dir_path = os.path.join(self.out_dir_path, 'readcounts')
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)
        self.review = pd.DataFrame(columns=['chromosome', 'start', 'stop',
                                            'ref', 'var', 'call', 'reviewer'])
        for sample in self.samples:
            print(os.getcwd())
            sites_file_path = os.path.join(out_dir_path, sample[0]+ '.sites')
            review = self._parse_review_file(sample[3], sites_file_path)
            self.review = pd.concat([self.review, review], ignore_index=True)
            tumor_readcount_file_path = '{0}/{1}_tumor'\
                '.readcounts'.format(out_dir_path, sample[0])
            os.system('bam-readcount -i -w 0 -l {0} -f {1} {2} > {3}'.format(
                sites_file_path, sample[5],sample[1],
                tumor_readcount_file_path))
            normal_readcount_file_path = '{0}/{1}_normal' \
                                        '.readcounts'.format(out_dir_path,
                                                             sample[0])
            os.system('bam-readcount -i -w 0 -l {0} -f {1} {2} > {3}'.format(
                sites_file_path, sample[5], sample[2],
                normal_readcount_file_path))


    def _parse_review_file(self, manual_review_file_path, sites_file_path):
        manual_review = pd.read_csv(manual_review_file_path, sep='\t',
                                    names=['chromosome', 'start', 'stop',
                                            'ref', 'var', 'call', 'reviewer'])
        manual_review = manual_review.apply(self._convert_one_based, axis=1)
        manual_review = manual_review.replace('', np.nan).dropna(how='all')
        manual_review[['chromosome', 'start', 'stop']].to_csv(sites_file_path,
                                                              sep='\t',
                                                              index=False,
                                                              header=False)
        return manual_review

    def _convert_one_based(self, row):
        return convert.coordinate_system('\t'.join(map(str, row.values)),
                                         'to_one_based').split('\t')
