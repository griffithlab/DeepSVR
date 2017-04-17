import re
import pandas as pd
from manual_review_classifier.utils import to_numeric


BASE_METRICS = ['count', 'avg_mapping_quality', 'avg_basequality',
                'avg_se_mapping_quality', 'num_plus_strand',
                'num_minus_strand', 'avg_pos_as_fraction',
                'avg_num_mismaches_as_fraction', 'avg_sum_mismatch_qualities',
                'num_q2_containing_reads',
                'avg_distance_to_q2_start_in_q2_reads',
                 'avg_clipped_length', 'avg_distance_to_effective_3p_end']

class ReadCount:
    """Parse bam-readcount out into dict or pandas.Dataframe
    
    """

    def __init__(self, file_path):
        """Initialize dict of bam-readcount file.
        
        Args:
            file_path (str): File path of bam-readcount file  
        """
        self.read_count_dict = self._parse(file_path)
        self.read_count_df = pd.DataFrame()

    def _parse(self, file_path):
        """Read and parse the bam_readcount file into a dict
        
        Args:
            file_path (str): File path of bam-readcount file  
        
        Returns:
            dict of bam-readcount
        """
        file = open(file_path)
        counts = {}
        for line in file:
            match = re.match(r'(^[\w|.]+\t\d+\t\w\t\d+)', line)
            if match is not None:
                count = re.split('\t|:', line.strip())
                position = '{0}:{1}'.format(count[0], count[1])
                metrics = {}
                metrics['chromosome'] = count[0]
                metrics['position'] = int(count[1])
                metrics['ref'] = count[2]
                metrics['depth'] = int(count[3])
                bases = {}
                for i in range(4, len(count), 14):
                    b = list(map(to_numeric, count[i + 1: i + 14]))
                    # if not all(x == 0 for x in b):
                    if count[i] != '=':
                        bases[count[i]] = dict(zip(BASE_METRICS, b))
                metrics['bases'] = bases
                counts[position] = metrics
        return counts

    def compute_variant_metrics(self, var_bed_file_path, sample_prepend_string):
        """
        
        Args:
            var_bed_file_path (str): filepath to bed file indicating variants
        
        Returns:
             pandas.Dataframe of variant centric read count data 
        """
        # TODO write code that specifically handles snvs and indels
        if len(self.read_count_df) > 0:
            return self.read_count_df
        with open(var_bed_file_path, 'r') as f:
            for line in f:
                debug = line.strip()
                line = line.strip().split('\t')
                chromosome, start, stop, reference, variant = line
                start = int(start)
                stop = int(stop)
                site = '{0}:{1}'.format(chromosome, start)

                # insersions
                if reference == '-':
                    self.flatten_base_metrics(self.read_count_dict[site]['ref'], site, sample_prepend_string+'_ref')
                    self.flatten_base_metrics('+{0}'.format(variant), site, sample_prepend_string+'_var')
                    # This dropping the counts at other positions could eliminate some real signal
                    self.remove_extra_indel_counts(chromosome, start, stop)
                # deletions
                elif variant == '-':
                    self.flatten_base_metrics(self.read_count_dict[site]['ref'], site, sample_prepend_string+'_ref')
                    self.flatten_base_metrics('-{0}'.format(reference), site, sample_prepend_string+'_var')
                    self.remove_extra_indel_counts(chromosome, start, stop)
                # snvs
                else:
                    self.flatten_base_metrics(reference, site, sample_prepend_string+'_ref' )
                    self.flatten_base_metrics(variant, site, sample_prepend_string+'_var')
                other_bases_count = 0
                for base in self.read_count_dict[site]['bases']:
                    #TODO check if other base metrics are nearly identical due to proximity
                    other_bases_count += self.read_count_dict[site]['bases'][base]['count']
                self.read_count_dict[site][sample_prepend_string+'_other_bases_count'] = other_bases_count
                self.read_count_dict[site].pop('bases')
                self.read_count_dict[site]['ref'] = reference
                self.read_count_dict[site]['var'] = variant
                self.read_count_dict[site]['stop'] = stop
                self.read_count_dict[site]['start'] = self.read_count_dict[site].pop('position')
                self.read_count_dict[site][sample_prepend_string+'_depth'] = self.read_count_dict[site].pop('depth')
        # print(self.read_count_dict)
        self.read_count_df = pd.DataFrame.from_dict(self.read_count_dict,orient='index')
        return self.read_count_df

    def remove_extra_indel_counts(self, chromosome, start, stop):
        for i in range(start + 1, stop +1 ):
            remove_site = '{0}:{1}'.format(chromosome, i)
            self.read_count_dict.pop(remove_site)

    def flatten_base_metrics(self, base_key, site, prepend_string):
        """ Renames the base metrics and flatten the data into the site level
        
        :param base_key (str): Key for the reference or variant 
        :param site (str): Site string 
        :param prepend_string (str): string to prepend the new keys
        """

        try:
            for metric_key in self.read_count_dict[site]['bases'][base_key]:
                new_key = '{0}_{1}'.format(prepend_string, metric_key)
                self.read_count_dict[site][new_key] = \
                    self.read_count_dict[site]['bases'][base_key][metric_key]
            # Remove ref or var metrics so counts for other bases can be summed
            self.read_count_dict[site]['bases'].pop(base_key)
        except KeyError:
            # add all zero metrics for indels absent from count file
            keys = ['{0}_{1}'.format(prepend_string, i) for i in BASE_METRICS]
            self.read_count_dict[site].update(dict.fromkeys(keys, 0))
