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

