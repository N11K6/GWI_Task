#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 22:16:12 2025

@author: nk
"""
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

def perform_HDBSCAN(config, data_reduced):
    min_cluster_size = int(config['CLUSTERING']['min_cluster_size'])
    data_clustered = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(data_reduced)
    return data_clustered

def plot2d(data_reduced, data_clustered):
    # Plot results
    plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=data_clustered, cmap='viridis', marker='o', s=50)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('2D Cluster Visualization')
    plt.colorbar(label='Cluster')
    plt.savefig('2d_cluster_visualization.png')
    return 0

def plot3d(data_reduced, data_clustered):
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
    plt.savefig('3d_cluster_visualization.png')
    return 0

def perform_clustering(config, dataset, data_reduced):
    data_clustered = perform_HDBSCAN(config,data_reduced)
    if data_reduced.shape[1] == 2:
        plot2d(data_reduced, data_clustered)
    elif data_reduced.shape[1] > 2:
        plot3d(data_reduced, data_clustered)
    return data_clustered