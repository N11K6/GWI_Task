#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for Clustering with HDBSCAN & getting visualizations etc

@author: nk
"""
import os
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

def perform_HDBSCAN(config, data_reduced):
    min_cluster_size = int(config['CLUSTERING']['min_cluster_size'])
    data_clustered = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(data_reduced)
    return data_clustered

def plot2d(data_reduced, data_clustered, output_dir=''):
    # Plot results
    plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=data_clustered, cmap='viridis', marker='o', s=50)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('2D Cluster Visualization')
    plt.colorbar(label='Cluster')
    plt.savefig(os.path.join(output_dir, '2d_cluster_visualization.png'))
    plt.close()

    return 0

def plot3d(data_reduced, data_clustered, output_dir=''):
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot with different colors for each cluster
    scatter = ax.scatter(
        xs=data_reduced[:, 0],  # PC1
        ys=data_reduced[:, 1],  # PC2
        zs=data_reduced[:, 2],  # PC3
        c=data_clustered,      # Cluster labels
        cmap='viridis',  # Color map
        s=50,           # Marker size
        alpha=0.8,      # Transparency
        depthshade=True # Better depth perception
    )
    # Add labels
    ax.set_xlabel('Component 1', fontsize=12, labelpad=10)
    ax.set_ylabel('Component 2', fontsize=12, labelpad=10)
    ax.set_zlabel('Component 3', fontsize=12, labelpad=10)
    ax.set_title('3D Cluster Visualization', fontsize=14, pad=20)
    # Add color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Cluster', fontsize=12)
    # Adjust viewing angle (elevation, azimuth)
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_cluster_visualization.png'))
    plt.close()

    return 0

def perform_clustering(config, data_reduced):\
    # Quick check to make sure we have data
    if data_reduced.size == 0:
        raise ValueError('Dataset after feature reduction is empty!')
    # Get temp output directory
    output_dir = config['TEMP']['temp_dir']
    
    # Perform clustering
    logger.info('Performing clustering using HDBSCAN.')
    data_clustered = perform_HDBSCAN(config,data_reduced)
    
    # Plot clustered data
    if data_reduced.shape[1] == 2:
        plot2d(data_reduced, data_clustered,output_dir)
    elif data_reduced.shape[1] > 2:
        plot3d(data_reduced, data_clustered,output_dir)
        
    # Calculate clustering evaluation metrics
    sil_score = silhouette_score(data_reduced, data_clustered)
    logger.info(f"Silhouette Score: {sil_score}")
    db_score = davies_bouldin_score(data_reduced, data_clustered)
    logger.info(f"Davies-Bouldin Index: {db_score}")
    ch_score = calinski_harabasz_score(data_reduced, data_clustered)
    logger.info(f"Calinski-Harabasz Index: {ch_score}")
    
    
    # Generate dictionary with clustering evaluation metrics
    metrics_dict = {}
    metrics_dict['sil_score'] = sil_score
    metrics_dict['db_score'] = db_score
    metrics_dict['ch_score'] = ch_score
    
    return data_clustered, metrics_dict
