# Notes on papers:
	
#### Germline:
1. Machine learning:
   * Deep Variant: Preprint of deep learning approach from Google to call germile variants. The last author of this study was the first author of GATK.
2. Statistical:
   * GATK: Well regarded germline variant caller from the Broad

#### Somatic:
1. Machine learning:
   * Mutation-seq: Study from BC cancer agency who compare results of random forest, Bayesian 
                additive regression tree, support vector machine and logistic regression
                models on a training set of 48 genomes containing 3369 candidate variants
                (1015 somatic and 2354 non somatic).
   * SNooPer: Random forrest approach trained on 40 samples
   * SomaticSeq: This software runs 5 somatic callers then uses a adaptively boosted decision 
            tree learner to create a classifier for predicting mutation statuses.
2. Statistical:
Typically in somatic variant calling a union of these callers is used.
   * Mutect: probabilistic -- use Bayesian modeling to estimate likely joint normal-tumor 
        genotype probabilities
   * SomaticSniper: probabilistic -- use Bayesian modeling to estimate likely joint 
               normal-tumor genotype probabilities
   * Strekla: probabilistic -- use Bayesian modeling to estimate likely joint 
         normal-tumor genotype probabilities
   * Varscan2: heuristic--relies on independent analysis of tumor and normal genomes 
            followed by a statistical Fisher’s Exact Test of read counts for 
            variant detection

* Comparing somatic callers: Study on how to better compare somatic callers
* Comparative Analysis of Somatic Detection 2013: Review of somatic variant callers.
	
[Review of various callers](https://omictools.com/somatic-snp-detection-category)