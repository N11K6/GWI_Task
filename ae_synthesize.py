#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 17:28:07 2025

@author: nk
"""
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def prepare_autoencoder(X_scaled: np.ndarray, y_scaled: np.ndarray, encoding_dim=2):
    '''
    Build/Train/Save a simple autoencoder (CPU-friendly architecture)
    Also save the encoder part, to use as an embedding generator.
    '''
    # 1. Build the autoencoder
    input_dim = X_scaled.shape[1]
    output_dim = y_scaled.shape[1]
    
    # Smaller architecture for CPU training
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu')(input_layer)
    encoder = Dense(32, activation='relu')(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)  # Bottleneck
    
    decoder = Dense(32, activation='relu')(encoder)
    decoder = Dense(64, activation='relu')(decoder)
    decoder = Dense(output_dim, activation='sigmoid')(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # 2. Train with fewer epochs and larger batch size (better for CPU)
    autoencoder.fit(X_scaled, y_scaled,
                    epochs=50,  # Reduced from 100
                    batch_size=64,  # Increased from 32
                    shuffle=True,
                    validation_split=0.2,
                    verbose=1)
    # 3. Save the full autoencoder
    autoencoder.save('autoencoder_for_data_synthesis.keras')
        
    # 4. Done
    print('Autoencoder saved')
    
    return 0

def use_encoder(X_scaled):
    # Load the saved encoder
    loaded_autoencoder = load_model('autoencoder_for_data_synthesis.keras')
    
    # Use it for dimensionality reduction
    results = loaded_autoencoder.predict(X_scaled)
    
    return results

def prepare_data(dataset: pd.DataFrame) -> np.ndarray:  
    dataset_train = dataset.dropna()
    dataset_missing = dataset[dataset.isna().any(axis=1)]
    re_cols_lst = [col for col in dataset.columns if 'core_re' in col]
    dataset_X = dataset_train.drop(re_cols_lst, axis=1)
    dataset_y = dataset_train[re_cols_lst]    
    dataset_X_missing = dataset_missing.drop(re_cols_lst, axis=1)
    return dataset_X, dataset_y, dataset_X_missing

def synthesize_data(config, dataset):
    dataset_X, dataset_y, dataset_X_missing = prepare_data(dataset)
    #prepare_autoencoder(dataset_X, dataset_y, encoding_dim=4)
    results = use_encoder(dataset_X_missing)
    dataset_y_synthesized = pd.DataFrame(data = results, columns = dataset_y.columns)
    
    dataset_synthesized = pd.concat([dataset_X_missing.reset_index(drop=True), dataset_y_synthesized], axis=1)
    dataset_known = dataset.dropna(axis=0)
    dataset_full = pd.concat([dataset_known,dataset_synthesized]).reset_index(drop=True)
    return dataset_full

if __name__ == "__main__":
    dataset = pd.read_excel('dataset.xlsx')
    config = None
    dataset_full = synthesize_data(config, dataset)