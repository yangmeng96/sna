import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def pca_load(datatype, disease_mapping, X_train, X_test): # datatype: "binary" or "cont"
    # load csv files
    file_paths = ['data/pca/' + datatype + '/' + name + '/' for name in ["train", "test"]]
    for file_path in file_paths:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
    # only keep demographic covariates at the beginning
    pattern = re.compile(r'^[A-Z0-9]{3}$')
    demo_columns = [col for col in X_test.columns if not pattern.match(col)]
    X_train_pca = X_train[demo_columns].reset_index(drop=True) 
    X_test_pca = X_test[demo_columns].reset_index(drop=True)
    
    # for each disease group
    for disease, codes in disease_mapping.items():
        try:
            X_train_single = pd.read_csv(file_paths[0] + disease + ".csv")
            X_test_single = pd.read_csv(file_paths[1] + disease + ".csv")
            print(disease + " PCA data loaded")
        except:
            print(disease + " PCA data not found, training PCA now...")
            X_train_single, X_test_single = pca_train(datatype, disease, codes, file_paths, X_train, X_test)
    
        # attach pca results back to demographic data
        X_train_pca = pd.concat(
            [X_train_single, X_train_pca], axis=1)            
        X_test_pca = pd.concat(
            [X_test_single, X_test_pca], axis=1) 
        
    return X_train_pca, X_test_pca

def pca_train(datatype, disease, codes, file_paths, X_train, X_test):
    
    # only apply pca to given disease group
    X_train, X_test = X_train.loc[:, codes], X_test.loc[:, codes]

    # Train PCA on the Training Set
    # Standardize the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit PCA on the standardized training data
    pca = PCA(n_components=None)  # You can choose the number of components or leave it as None
    pca.fit(X_train_scaled)
    explained_variance = pca.explained_variance_ratio_

    # Apply the Trained PCA on Both Sets
    X_test_scaled = scaler.transform(X_test)
    X_train_pca = pd.DataFrame(pca.transform(X_train_scaled))
    X_test_pca = pd.DataFrame(pca.transform(X_test_scaled))

    # rename to PC0, PC1, PC2, etc.
    X_train_pca.rename(columns=lambda x: disease+"_PC"+str(x), inplace=True)
    X_test_pca.rename(columns=lambda x: disease+"_PC"+str(x), inplace=True)

    # save csv files
    train_csv = file_paths[0] + disease + ".csv"
    test_csv = file_paths[1] + disease + ".csv"
    X_train_pca.to_csv(train_csv, index=False)
    print(disease + " PCA training data saved as: " + train_csv)
    X_test_pca.to_csv(test_csv, index=False)
    print(disease + " PCA test data saved as: " + test_csv)

    # visualize cumulative explained variance
    plot_cum_var_bar(explained_variance, datatype, disease)

    return X_train_pca, X_test_pca

def plot_cum_var_bar(explained_variance, datatype, disease):
    # load figure
    file_path = 'figs/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(10, 6))
    plt.bar(x=range(1, len(cumulative_explained_variance)+1), 
            height=cumulative_explained_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(disease + ' - Cumulative Explained Variance by PCA Components')

    for num_pc in range(len(cumulative_explained_variance)):
        plt.text(num_pc+1-.12, cumulative_explained_variance[num_pc]+.01, 
                 str(int(cumulative_explained_variance[num_pc]*100))+"%", 
                 color = 'black', fontsize=12)

    plt.savefig(file_path + datatype + "_" + disease + "_pca.png") 
    plt.show()