import re
from manual_review_classifier.utils import to_numeric

class ReadCount:
    """Parse bam-readcount out into dict or pandas.Dataframe
    
    """

    def __init__(self, file_path):
        """Initialize dict of bam-readcount file.
        
        Args:
            file_path (str): File path of bam-readcount file  
        """
        self.read_count_dict = self._parse(file_path)


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
            match = re.match(r'(^\w+\t\d+\t\w\t\d+)', line)
            if match is not None:
                count = re.split('\t|:', line.strip())
                position = '{0}:{1}{2}'.format(count[0],
                                               count[1],
                                               count[2])
                metrics = {}
                metrics['chromosome'] = count[0]
                metrics['position'] = int(count[1])
                metrics['ref'] = count[2]
                metrics['depth'] = int(count[3])
                bases = {}
                for i in range(4, len(count), 14):
                    base_metrics = ['count', 'avg_mapping_quality'
                                             'avg_basequality',
                                    'avg_se_mapping_quality',
                                    'num_plus_strand', 'num_minus_strand',
                                    'avg_pos_as_fraction',
                                    'avg_num_mismaches_as_fraction',
                                    'avg_sum_mismatch_qualities',
                                    'num_q2_containing_reads',
                                    'avg_distance_to_q2_start_in_q2_reads',
                                    'avg_clipped_length',
                                    'avg_distance_to_effective_3p_end']
                    b = list(map(to_numeric, count[i + 1: i + 14]))
                    if not all(x == 0 for x in b):
                        bases[count[i]] = dict(zip(base_metrics, b))
                metrics['bases'] = bases
                counts[position] = metrics
        return counts

