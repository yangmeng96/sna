import pandas as pd
import numpy as np
import os
import re

def baseline(datatype, disease_mapping, X_train, X_test): # datatype: "binary" or "cont"

    # only keep demographic covariates at the beginning
    pattern = re.compile(r'^[A-Z0-9]{3}$')
    demo_columns = [col for col in X_test.columns if not pattern.match(col)]
    X_train_max = X_train[demo_columns].reset_index(drop=True) 
    X_test_max = X_test[demo_columns].reset_index(drop=True)
    
    # max aggregation
    for disease, codes in disease_mapping.items():
        X_train_single = X_train.loc[:, codes].max(axis=1).to_frame().rename(
            columns={0:disease}).reset_index(drop=True)
        X_test_single = X_test.loc[:, codes].max(axis=1).to_frame().rename(
            columns={0:disease}).reset_index(drop=True)
        X_train_max = pd.concat(
            [X_train_single, X_train_max], axis=1)            
        X_test_max = pd.concat(
            [X_test_single, X_test_max], axis=1) 
    return X_train_max, X_test_max