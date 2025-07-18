#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 19:27:23 2025

@author: nk
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#%%
def prepare_data(dataset: pd.DataFrame) -> np.ndarray:
    dataset.dropna(inplace=True)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(dataset)
    return X_scaled

def prepare_autoencoder(X_scaled: np.ndarray, encoding_dim=2):
    '''
    Build/Train/Save a simple autoencoder (CPU-friendly architecture)
    Also save the encoder part, to use as an embedding generator.
    '''
    # 1. Build the autoencoder
    input_dim = X_scaled.shape[1]
    
    # Smaller architecture for CPU training
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu')(input_layer)
    encoder = Dense(32, activation='relu')(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)  # Bottleneck
    
    decoder = Dense(32, activation='relu')(encoder)
    decoder = Dense(64, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # 2. Train with fewer epochs and larger batch size (better for CPU)
    autoencoder.fit(X_scaled, X_scaled,
                    epochs=50,  # Reduced from 100
                    batch_size=64,  # Increased from 32
                    shuffle=True,
                    validation_split=0.2,
                    verbose=1)
    # 3. Save the full autoencoder
    autoencoder.save('autoencoder_model.keras')
    
    # 4. Create and save just the encoder
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    encoder_model.save('encoder_model.keras')
    
    # 5. Done
    print('Autoencoder saved')
    print('Encoder saved.')
    
    return 0

def use_encoder(X_scaled):
    # Load the saved encoder
    loaded_encoder = load_model('encoder_model.keras')
    
    # Use it for dimensionality reduction
    encoded_data = loaded_encoder.predict(X_scaled)
    
    return encoded_data

#%%
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

def perform_clustering(X):
    clusters = HDBSCAN(min_cluster_size=20).fit_predict(X)
    return clusters

def plot2d(X_pca, clusters):
    # Plot results
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', marker='o', s=50)
    plt.xlabel('Embedding dimension 1')
    plt.ylabel('Embedding dimension 2')
    plt.title('Encoder + HDBSCAN Clustering')
    plt.colorbar(label='Cluster Label')
    plt.show()
    return 0

def plot3d(X_pca, clusters):
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot with different colors for each cluster
    scatter = ax.scatter(
        xs=X_pca[:, 0],  # PC1
        ys=X_pca[:, 1],  # PC2
        zs=X_pca[:, 2],  # PC3
        c=clusters,      # Cluster labels
        cmap='viridis',  # Color map
        s=50,           # Marker size
        alpha=0.8,      # Transparency
        depthshade=True # Better depth perception
    )
    # Add labels
    ax.set_xlabel('Principal Component 1', fontsize=12, labelpad=10)
    ax.set_ylabel('Principal Component 2', fontsize=12, labelpad=10)
    ax.set_zlabel('Principal Component 3', fontsize=12, labelpad=10)
    ax.set_title('3D Cluster Visualization', fontsize=14, pad=20)
    # Add color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Cluster', fontsize=12)
    # Adjust viewing angle (elevation, azimuth)
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    plt.show()
    
    return 0

#%%
if __name__ == "__main__":
    dataset = pd.read_excel('dataset.xlsx')
    X_scaled = prepare_data(dataset)
    prepare_autoencoder(X_scaled, encoding_dim=4)
    encoded_data = use_encoder(X_scaled)
    clusters = perform_clustering(encoded_data)
    plot3d(encoded_data,clusters)
    #plot2d(encoded_data,clusters)
    print('ok')