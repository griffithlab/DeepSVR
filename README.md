# manual_review_classifier

### Analysis

All analysis is viewable in the `notebooks` directory

### Installation of prepare data script

This script will run bam-readcount on bam files and assemble a pandas dataframe ready to be 
used in machine learning as is done in `notebooks/Deep learning model.ipynb`.

A tab separated sample file is required as input that outlines sample_name,
tumor_bam_path, normal_bam_path, manual_review_file_path, reviewer, disease,
and reference_genome_fasta_file_path. Use `-h` for mor information.

Install the package using the editable (`-e` option) mode for ease in development

`git clone [manual_review_classifier_link]`

`cd manual_review_classifier`

`pip install -e .`

Test instalation 

`prepare_manual_review_classifier_data -h`

### Test

To run tests Download GRCH37 fasta and fasta index file to home directory as 
'\~/grch37.fa' and '\~/grch37.fa.fai'
