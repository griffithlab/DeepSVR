# Create Classifier
USAGE: python3 CreateClassifier.py <training data path>

EXAMPLE: python3 CreateClassifier.py /data/training_data.pkl

<training data path> = the training data path should point to a .pkl file that has the following labels:
  ['call', 'normal_VAF', 'normal_depth', 'normal_other_bases_count', 'normal_ref_avg_basequality', 'normal_ref_avg_clipped_length',
 'normal_ref_avg_distance_to_effective_3p_end', 'normal_ref_avg_distance_to_q2_start_in_q2_reads', 'normal_ref_avg_mapping_quality',
 'normal_ref_avg_num_mismaches_as_fraction', 'normal_ref_avg_pos_as_fraction', 'normal_ref_avg_se_mapping_quality',
 'normal_ref_avg_sum_mismatch_qualities', 'normal_ref_count', 'normal_ref_num_minus_strand', 'normal_ref_num_plus_strand',
 'normal_ref_num_q2_containing_reads', 'normal_var_avg_basequality', 'normal_var_avg_clipped_length',
 'normal_var_avg_distance_to_effective_3p_end', 'normal_var_avg_distance_to_q2_start_in_q2_reads', 'normal_var_avg_mapping_quality',
 'normal_var_avg_num_mismaches_as_fraction', 'normal_var_avg_pos_as_fraction', 'normal_var_avg_se_mapping_quality',
 'normal_var_avg_sum_mismatch_qualities', 'normal_var_count', 'normal_var_num_minus_strand', 'normal_var_num_plus_strand',
 'normal_var_num_q2_containing_reads', 'tumor_VAF', 'tumor_depth', 'tumor_other_bases_count', 'tumor_ref_avg_basequality',
 'tumor_ref_avg_clipped_length', 'tumor_ref_avg_distance_to_effective_3p_end', 'tumor_ref_avg_distance_to_q2_start_in_q2_reads',
 'tumor_ref_avg_mapping_quality', 'tumor_ref_avg_num_mismaches_as_fraction', 'tumor_ref_avg_pos_as_fraction',
 'tumor_ref_avg_se_mapping_quality', 'tumor_ref_avg_sum_mismatch_qualities', 'tumor_ref_count', 'tumor_ref_num_minus_strand',
 'tumor_ref_num_plus_strand', 'tumor_ref_num_q2_containing_reads', 'tumor_var_avg_basequality', 'tumor_var_avg_clipped_length',
 'tumor_var_avg_distance_to_effective_3p_end', 'tumor_var_avg_distance_to_q2_start_in_q2_reads', 'tumor_var_avg_mapping_quality', 
 'tumor_var_avg_num_mismaches_as_fraction', 'tumor_var_avg_pos_as_fraction','tumor_var_avg_se_mapping_quality',
 'tumor_var_avg_sum_mismatch_qualities', 'tumor_var_count', 'tumor_var_num_minus_strand', 'tumor_var_num_plus_strand',
 'tumor_var_num_q2_containing_reads', 'solid_tumor']




# Classifier Data
USAGE: python3 cli.py  <header?> <bam-readcounts?> <sample file path> <solid tumor?> <output path>

EXAMPLE: python3 cli.py --no-header --no-skip_bam_readcount --samples-file-path /Users/ebarnell/manual_review_classifier/manual_review_classifier/tests/test_data/samples.noheader.tsv  --solid_tumor --output-dir-path output/

<header?> = if the sample file path has a head put --header else put --no-header

<bam-readcounts?> = if the samples in the sample file path require bam readcounds to be run (i.e. this is the first time you are running your samples through the classifier) then put --no-skip_bam_readcount else put --skip_bam_readcount

<sample file path> = sample file path should direct you to a tab-separated file with the following columns:
  [sample_name	tumor_bam	normal_bam	manual_review	reviewer	disease	reference_fasta_file_path]
  sample_name = unique identifier
  tumor_bam = path for *.bam file to pull the bam-readcounts for the tumor sample
  normal_bam = path for *.bam file to pull the bam-readcounts for the normal sample
  manual_review = .BED or .BED-like file that has chromosome, start, stop, ref, var for all variants requiring classification
  reviewer = put your individual name
  disease = put disease of sample being analyzed
  reference_fasta_file_path = point to where the GCRH37.fa is housed on your computer (you need to download this remotely)

<solid tumor> = if the samples being analyzed are solid tumors, put --solid_tumor else put --no-solid_tumor
  
<output path> = create an output path for your files


