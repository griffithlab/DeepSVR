# DeepSVR
This repository can be used to recapitulate the development and analysis of a machine learning model approach to somatic variant refinement. DeepSVR contains the raw data (e.g. bam files and manual review labels), code required for data preparation, and validation sets to test the ultimate models. Using the prepared data, we developed three machine learning models (Logistic Regression, Random Forest, and feed-forward Deep Learning). The model that was most consistent with manual revew labels was the Deep Learning model. This model was packaged and is available for use.

### A walk-through of the DeepSVR repo can be found on the [Wiki page](https://github.com/griffithlab/manual_review_classifier/wiki). 


### Installation of deepsvr package
*Note: Please ensure that you are running these commands using python3 or greater.*

#### 1) Clone the DeepSVR GitHub Repo see [Repository - Installation](https://github.com/griffithlab/DeepSVR/wiki/Repository-Installation)

#### 2) Install Anaconda see [Downloads - Anaconda](https://www.anaconda.com/download/)

#### 3) Add BioConda Channels
`conda config --add channels defaults`  
`conda config --add channels conda-forge` 
`conda config --add channels bioconda`  

#### 4) Install DeepSVR see [BioConda - DeepSVR](https://anaconda.org/bioconda/deepsvr)
`conda install deepsvr`

#### 5) Test installation and view DeepSVR options
`deepsvr --help`
