# DeepSVR
This repository can be used to recapitulate the development and analysis of a machine learning model approach to somatic variant refinement. DeepSVR contains the raw data (e.g. bam files and manual review labels), code required for data preparation, and validation sets to test the ultimate models. Using the prepared data, we developed three machine learning models (Logistic Regression, Random Forest, and feed-forward Deep Learning). The model that was most consistent with manual revew labels was the Deep Learning model. This model was packaged and is available for use.

### A walk-through of the DeepSVR repo can be found on the [Wiki page](https://github.com/griffithlab/manual_review_classifier/wiki). 

### Installation of DeepSVR

1) Compile and install [bam-readcount](https://github.com/genome/bam-readcount#build-instructions)
2) Install DeepSVR with pip:  `pip install deepsvr`

### Test

To run tests Download GRCH37 fasta and fasta index file to home directory as 
'\~/grch37.fa' and '\~/grch37.fa.fai'
