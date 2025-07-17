# -*- coding: utf-8 -*-
"""
GWI Technical Task
Author: Nikos Kournoutos

ROUGH CODE FOR TESTING
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

def read_data(filepath):
    df = pd.read_excel(filepath)

    n_subjects = df.shape[0]
    n_features = df.shape[1]
    n_total_entries = n_subjects*n_features
    nan_count = df.isna().sum().sum()
    
    print(f'{nan_count} NaN entries out of {n_total_entries} total entries \
          ({np.round(100*nan_count/n_total_entries)}%)')
    
    return df


def feature_reduction(df):
    
    # Ignore missing values
    missing_cols = []
    for col in df.columns:
        missing_mask = df[col].isna()
        if missing_mask.any():
            missing_cols.append(col)
    
    df.drop(missing_cols,axis=1,inplace=True)
    
    # Variance check
    df = df.loc[:,df.var()>0.05]
    
    # Correlation check
    corr_matrix = df.corr().abs()
    threshold = 0.5
    high_corr = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                colname = corr_matrix.columns[i]
                high_corr.add(colname)
    
    df_reduced = df.drop(columns=high_corr)    
    
    return df_reduced


def perform_pca(df):
    # PCA
    pca = PCA(n_components=None)
    X_pca = pca.fit_transform(df)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot the explained variance
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Components')
    plt.grid()
    plt.show()
    
    # Choose number of components where adding more doesn't give much improvement
    # Common thresholds are 80%, 90%, or 95% variance explained
    desired_variance = 0.999
    optimal_n_components = len(np.argwhere(cumulative_variance >= desired_variance)) + 1
    
    pca_optimal = PCA(n_components=optimal_n_components)
    print(f'optimal components: {optimal_n_components}')
    X_pca = pca_optimal.fit_transform(df)
    
    return X_pca


def perform_clustering(X):
    clustering = HDBSCAN(min_cluster_size=50).fit_predict(X)
    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=clustering, cmap='viridis', marker='o', s=50)
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.title('PCA + DBSCAN Clustering (Binary Data)')
    plt.colorbar(label='Cluster Label')
    plt.show()
    #Silhoutte score to evaluate clusters
    print(silhouette_score(X, clustering))
    return clustering

if __name__ == "__main__":
    df = read_data('dataset.xlsx')
    df = feature_reduction(df)
    X_pca = perform_pca(df)
    
    clustering = perform_clustering(X_pca)
    