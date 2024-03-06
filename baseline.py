import pandas as pd
import numpy as np
import os
import re

def baseline(datatype, disease_mapping, agg_method, X_train, X_test): 
    # datatype: "binary" or "cont"
    # agg_method: "max" or "sum"

    # only keep demographic covariates at the beginning
    pattern = re.compile(r'^[A-Z0-9]{3}.*')
    demo_columns = [col for col in X_test.columns if not pattern.match(col)]
    X_train_agg = X_train[demo_columns].reset_index(drop=True) 
    X_test_agg = X_test[demo_columns].reset_index(drop=True)
    
    # aggregation
    for disease, codes in disease_mapping.items():
        
        regex_pattern = '|'.join(f'^{code}' for code in codes)
        
        if agg_method == "max":
            
            X_train_single = X_train.filter(
                regex=regex_pattern, axis=1).max(axis=1).to_frame().rename(
                columns={0:disease}).reset_index(drop=True)
            X_test_single = X_test.filter(
                regex=regex_pattern, axis=1).max(axis=1).to_frame().rename(
                columns={0:disease}).reset_index(drop=True)
        
        elif agg_method == "sum":
            
            X_train_single = X_train.filter(
                regex=regex_pattern, axis=1).sum(axis=1).to_frame().rename(
                columns={0:disease}).reset_index(drop=True)
            X_test_single = X_test.filter(
                regex=regex_pattern, axis=1).sum(axis=1).to_frame().rename(
                columns={0:disease}).reset_index(drop=True)
            
        X_train_agg = pd.concat(
            [X_train_single, X_train_agg], axis=1)            
        X_test_agg = pd.concat(
            [X_test_single, X_test_agg], axis=1) 
        
    return X_train_agg, X_test_agg