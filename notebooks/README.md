# Notebooks

## Preprocessing notebooks
These notebooks are run in the order listed to prepare the training data.

Before running these notebooks please unzip the raw bam-readcount files in
`../data/bam-read-counts/normal.zip`. A folder named `normal` should be in the 
`../data/bam-read-counts/` directory.

Note: running these notebooks is not necessary to run the machine learning 
models. The result of these notebooks, `../data/training_data.pkl`, is available
in the repository.

### 1. Add project specific features.ipynb
Add disease and reviewer information to analysis

### 2. Duplicate analysis.ipynb
Preform pairwise comparisons between mutations in every sample to ensure that 
samples that are duplicated between projects with not confound subsequent 
analysis.

### 3. parse bam read count.ipynb

This notebook contains the parsing of the bam-readcount files along with the 
initial pre-processing of the training dataset.

## Model building notebooks

### Deep learning model.ipynb

This notebook contains the analysis of the deep learning model.

### Logistic regression classifier.ipynb

This notebook contains the
 analysis of the logistic regression model.

