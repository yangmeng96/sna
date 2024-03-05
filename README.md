# Stats 295 Final

This is the repo of Xueting Ding, Yang Meng, Liner Xiang's Stats 295 final project.
 
## To run the experiments:
 
1. Put the raw data (a ".sas7bdat" file) into /data path

2. Run "run_experiments.ipynb", uncomment the first cell to preprocess the data

## baseline.py

Given the mapping of "a big disease group: a few codes", we pick one covariate/ code from the disease group using maximum aggregation. 

## pca.py

Given the mapping of "a big disease group: a few codes", we apply PCA to this group and pick the top components in the downstream tasks.

## pca_all.py

We apply PCA to all the disease-related covariates and pick the top components in the downstream tasks.

## lr.py

Build Logistic regression model.

## rf.py

Build Random Forest classifier.
