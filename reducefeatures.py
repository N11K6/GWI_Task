#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for feature/dimensionality reduction

@author: nk
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import logging
logger = logging.getLogger(__name__)
import ae_reduce

def perform_varcorrcheck(config, dataset):
    # Get temp output directory
    output_dir = config['TEMP']['temp_dir']
    
    # Variance check
    if int(config['PROCESSING']['variance_check']) == 1:
        variance_threshold = float(config['PROCESSING']['variance_threshold'])
        logger.info(f'Performing variance check with threshold {variance_threshold}.')
        num_features_before = dataset.shape[1]
        dataset = dataset.loc[:,dataset.var()>variance_threshold]
        dataset.var().to_csv(os.path.join(output_dir, 'dataset_variance.csv'),index=False)
        logger.info(f'{num_features_before-dataset.shape[1]} features dropped due to low variance.')
        logger.info(f'{dataset.shape[1]} features remain.')

    # Correlation check
    if int(config['PROCESSING']['correlation_check']) == 1:
        correlation_threshold = float(config['PROCESSING']['correlation_threshold'])
        logger.info(f'Performing correlation check with threshold {correlation_threshold}.')
        num_features_before = dataset.shape[1]
        corr_matrix = dataset.corr().abs()
        corr_matrix.var().to_csv(os.path.join(output_dir, 'dataset_correlation_matrix.csv'),index=False)
        high_corr = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    colname = corr_matrix.columns[i]
                    high_corr.add(colname)
        dataset = dataset.drop(columns=high_corr)    
        logger.info(f'{num_features_before-dataset.shape[1]} features dropped due to high correlation.')
        logger.info(f'{dataset.shape[1]} features remain.')

    return dataset

def perform_pca(config, dataset):
    # Get temp output directory
    output_dir = config['TEMP']['temp_dir']
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
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance by Components')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'explained_variance.png'))
    plt.clf()
    # Plot the explained variance (normalized)
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(normalized_data)+1), normalized_data, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Normalized Explained Variance')
    plt.title('Explained Variance by Components, Normalized')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'explained_variance_normalized.png'))
    plt.clf()
    # Read desired variance (normalized) for PCA from config
    desired_variance = float(config['PROCESSING']['pca_desired_variance'])
    logger.info(f'Keeping components with (normalized) variance threshold at {desired_variance}')
    optimal_n_components = len(np.argwhere(normalized_data >= desired_variance))
    pca_optimal = PCA(n_components=optimal_n_components)
    logger.info(f'No. of optimal components: {optimal_n_components}')
    X_pca = pca_optimal.fit_transform(dataset)
    
    return X_pca

def reduce_features(config, dataset):
    dataset = perform_varcorrcheck(config, dataset)
    if config['PROCESSING']['dimensionality_reduction'].lower() == 'pca':
        logger.info('Performing dimensionality reduction using PCA')
        data_reduced = perform_pca(config, dataset)
    elif config['PROCESSING']['dimensionality_reduction'].lower() == 'encoder':
        logger.info('Performing Dimensionality reduction using Encoder')
        data_reduced = ae_reduce.perform_encoding(config, dataset)
    else:
        logger.warning(f'Configuration has not specified a valid method for dimensionality reduction.\
              \n Clustering will be attempted with {dataset.shape[1]} features')
        data_reduced = dataset.to_numpy() # convert to numpy for consistency
    
    return data_reduced
