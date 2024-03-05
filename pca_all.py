import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def pca_load_all(datatype, X_train, X_test): # datatype: "binary" or "cont"
    
    # load csv files
    file_path = 'data/pca/' + datatype + '/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        X_train_pca = pd.read_csv("data/pca/" + datatype + "/X_train.csv")
        X_test_pca = pd.read_csv("data/pca/" + datatype + "/X_test.csv")
        print("PCA data loaded")

    except:
        print("PCA data not found, running PCA now...")

        # only apply pca to illness covariates
        pattern = re.compile(r'^[A-Z0-9]{3}$')
        selected_columns = [col for col in X_test.columns if pattern.match(col)]
        unselected_columns = [col for col in X_test.columns if not pattern.match(col)]
        X_train_illness, X_test_illness = X_train[selected_columns], X_test[selected_columns]
        X_train_demo, X_test_demo = X_train[unselected_columns], X_test[unselected_columns]

        # Train PCA on the Training Set
        # Standardize the training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_illness)

        # Fit PCA on the standardized training data
        pca = PCA(n_components=None)  # You can choose the number of components or leave it as None
        pca.fit(X_train_scaled)
        explained_variance = pca.explained_variance_ratio_

        # Apply the Trained PCA on Both Sets
        X_test_scaled = scaler.transform(X_test_illness)
        X_train_pca = pca.transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # to pandas dataframe
        X_train_pca = pd.DataFrame(X_train_pca)
        X_test_pca = pd.DataFrame(X_test_pca)

        # rename to PC0, PC1, PC2, etc.
        X_train_pca.rename(columns=lambda x: "PC"+str(x), inplace=True)
        X_test_pca.rename(columns=lambda x: "PC"+str(x), inplace=True)

        # put demographic info back
        X_train_pca = pd.concat([X_train_pca, X_train_demo.reset_index(drop=True)], axis=1)
        X_test_pca = pd.concat([X_test_pca, X_test_demo.reset_index(drop=True)], axis=1)

        # save csv files
        X_train_pca.to_csv(file_path + "X_train.csv", index=False)
        print("PCA training data saved as: " + file_path + "X_train.csv")
        X_test_pca.to_csv(file_path + "X_test.csv", index=False)
        print("PCA test data saved as: " + file_path + "X_test.csv")
        
        # visualize cumulative explained variance
        plot_cum_var(explained_variance, datatype)

    return X_train_pca, X_test_pca

def plot_cum_var(explained_variance, datatype):
    # load figure
    file_path = 'figs/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_explained_variance, marker='', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by PCA Components')

    for cutoff in [.9, .8, .7, .6, .5]:
        plt.axhline(y=cutoff, color='r', linestyle='-')
        plt.text(0, cutoff+0.01, str(int(cutoff*100)) + '% cut-off threshold', color = 'red', fontsize=12)
        
    plt.savefig(file_path + datatype + "_pca.png") 
    plt.show()