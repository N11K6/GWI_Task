#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 20:48:56 2025

@author: nk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import dataload
import ae_reduce

def perform_varcorrcheck(config, dataset):
    # Variance check
    if int(config['PROCESSING']['variance_check']) == 1:
        variance_threshold = float(config['PROCESSING']['variance_threshold'])
        dataset = dataset.loc[:,dataset.var()>variance_threshold]
    
    # Correlation check
    if int(config['PROCESSING']['correlation_check']) == 1:
        correlation_threshold = float(config['PROCESSING']['correlation_threshold'])
        corr_matrix = dataset.corr().abs()
        high_corr = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    colname = corr_matrix.columns[i]
                    high_corr.add(colname)
        dataset = dataset.drop(columns=high_corr)    
        
    return dataset

def perform_pca(config, dataset):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataset)
    # PCA
    pca = PCA(n_components = "mle", svd_solver ="full")
    X_pca = pca.fit_transform(X_scaled)
    # Calculate cumulative explained variance
    cumulative_variance = pca.explained_variance_ratio_
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(cumulative_variance.reshape(-1, 1)).flatten()
    # Plot the explained variance
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(normalized_data)+1), normalized_data, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Normalized Explained Variance')
    plt.title('Explained Variance by Components, Normalized')
    plt.grid()
    plt.savefig('explained_variance_normalized.png')

    desired_variance = 0.5
    
    print(f'Keeping components with (normalized) variance threshold at {desired_variance}')
    optimal_n_components = len(np.argwhere(normalized_data >= desired_variance))
    
    pca_optimal = PCA(n_components=optimal_n_components)
    print(f'Optimal components: {optimal_n_components}')
    X_pca = pca_optimal.fit_transform(dataset)
    
    return X_pca

def reduce_features(config, dataset):
    dataset = perform_varcorrcheck(config, dataset)
    if config['PROCESSING']['dimensionality_reduction'].lower() == 'pca':
        print('Performing PCA')
        data_reduced = perform_pca(config, dataset)
    elif config['PROCESSING']['dimensionality_reduction'].lower() == 'encoder':
        print('Performing Dimensionality reduction using Encoder')
        data_reduced = ae_reduce.perform_encoding(config, dataset)
    else:
        print(f'Configuration has not specified a valid method for dimensionality reduction.\
              \n Clustering will be attempted with {dataset.shape[1]} features')
        data_reduced = dataset
    return data_reduced

if __name__ == "__main__":
    config = dataload.read_config()
    dataset = pd.read_excel('dataset_imputed.xlsx')
    data_reduced = reduce_features(config, dataset)