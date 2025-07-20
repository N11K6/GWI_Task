#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration testing for the Flask API

@author: nk
"""

import io
import configparser
import pandas as pd
import pytest
from appMain import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_get(client):
    """Test that the index page loads correctly"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'upload.html' in response.data  # Assuming your template has this string

def test_file_upload_success(client, tmp_path):
    """Test successful file upload and processing"""
    # Create test data
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3, None, 5],
        'feature2': [5, 4, None, 2, 1],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Create test config
    config = configparser.ConfigParser()
    config['MISSING'] = {
        'strategy': 'mean',
        'columns': 'feature1,feature2'
    }
    config['FEATURES'] = {
        'reduction_method': 'pca',
        'n_components': '2'
    }
    config['CLUSTERING'] = {
        'algorithm': 'kmeans',
        'n_clusters': '2'
    }
    
    # Prepare files for upload
    data_file = (io.BytesIO(), 'test_data.csv')
    test_data.to_csv(data_file[0], index=False)
    data_file[0].seek(0)
    
    ini_file = (io.BytesIO(), 'config.ini')
    config.write(ini_file[0])
    ini_file[0].seek(0)
    
    # Make the request
    response = client.post(
        '/',
        data={
            'data_file': data_file,
            'ini_file': ini_file
        },
        content_type='multipart/form-data'
    )
    
    # Verify response
    assert response.status_code == 200
    assert response.mimetype == 'application/zip'
    assert 'processed_results.zip' in response.headers['Content-Disposition']

def test_file_upload_missing_files(client):
    """Test when files are missing from the request"""
    response = client.post('/')
    assert response.status_code == 400
    assert b'Both data file and INI file are required' in response.data

def test_file_upload_invalid_types(client):
    """Test with invalid file types"""
    # Create dummy files with wrong extensions
    data_file = (io.BytesIO(b'dummy'), 'test.txt')
    ini_file = (io.BytesIO(b'dummy'), 'config.txt')
    
    response = client.post(
        '/',
        data={
            'data_file': data_file,
            'ini_file': ini_file
        },
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    assert b'Invalid file types' in response.data

def test_file_upload_empty_files(client):
    """Test with empty file selections"""
    data_file = (io.BytesIO(b''), 'test_data.csv')
    ini_file = (io.BytesIO(b''), 'config.ini')
    
    response = client.post(
        '/',
        data={
            'data_file': data_file,
            'ini_file': ini_file
        },
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    assert b'No selected files' in response.data
