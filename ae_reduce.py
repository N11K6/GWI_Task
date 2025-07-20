#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoEncoder for dimensionality reduction

@author: nk
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import logging
logger = logging.getLogger(__name__)

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
    logger.info('Training autoencoder for dimensionality reduction.')
    autoencoder.fit(X_scaled, X_scaled,
                    epochs=75,  # Reduced from 100
                    batch_size=64,  # Increased from 32
                    shuffle=True,
                    validation_split=0.2,
                    verbose=0)
    # 3. Save the full autoencoder
    autoencoder.save('autoencoder_for_reduction.keras')
    logger.info('Autoencoder ready.')

    # 4. Create and save just the encoder
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    encoder_model.save('encoder_for_reduction.keras')
    logger.info('Encoder ready.')
    
    return 0

def use_encoder(X_scaled):
    # Load the saved encoder
    loaded_encoder = load_model('encoder_for_reduction.keras')
    
    # Use it for dimensionality reduction
    logger.info('Using encoder for dimensionality reduction.')
    encoded_data = loaded_encoder.predict(X_scaled)
    
    return encoded_data

def perform_encoding(config, dataset):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(dataset)
    encoding_dim = int(config['PROCESSING']['encoding_dim'])
    logger.info(f'Encoding dimension: {encoding_dim}')
    prepare_autoencoder(data_scaled,encoding_dim)
    data_reduced = use_encoder(data_scaled)
    
    return data_reduced
