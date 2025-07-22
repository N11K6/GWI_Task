#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit testing for reducefeatures

@author: nk
"""
import pytest
import pandas as pd
import numpy as np
from reducefeatures import (
    perform_varcorrcheck,
    perform_pca,
    reduce_features
)
from unittest.mock import patch, MagicMock
import logging
import os

# Setup logging for testing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame for testing"""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 5, 5, 5, 5],  # Low variance column
        'C': [1, 2, 1, 2, 1],  
        'D': [10, 20, 30, 40, 50], # Correlated with A (to some degree)
        'E': [10, 10, 10, 10, 10]  # Low variance column
    })

@pytest.fixture
def sample_config():
    """Fixture providing a sample config dictionary"""
    return {
        'PROCESSING': {
            'variance_check': '1',
            'variance_threshold': '0.1',
            'correlation_check': '1',
            'correlation_threshold': '0.7',
            'dimensionality_reduction': 'pca',
            'pca_desired_variance': '0.95'
        },
        'TEMP': {
            'temp_dir': '/tmp'
        }
    }

@pytest.fixture
def mock_plt():
    """Fixture to mock matplotlib.pyplot"""
    with patch('matplotlib.pyplot') as mock_plt:
        yield mock_plt

def test_perform_varcorrcheck_variance_only(sample_dataframe, sample_config):
    """Test variance check without correlation check"""
    sample_config['PROCESSING']['correlation_check'] = '0'
    
    result = perform_varcorrcheck(sample_config, sample_dataframe)
        
    # Columns B and E should be dropped (variance <= 0.1)
    assert list(result.columns) == ['A', 'C', 'D']

def test_perform_varcorrcheck_correlation_only(sample_dataframe, sample_config):
    """Test correlation check without variance check"""
    sample_config['PROCESSING']['variance_check'] = '0'
    sample_config['PROCESSING']['correlation_threshold'] = '0.5'  # Lower threshold to catch correlation
    
    result = perform_varcorrcheck(sample_config, sample_dataframe)
        
    # Column C should be dropped (correlated with A)
    # We can't be sure which one will be dropped (A or C) as it depends on iteration order
    assert len(result.columns) == 4  # Either A or C remains

def test_perform_varcorrcheck_both_checks(sample_dataframe, sample_config):
    """Test both variance and correlation checks"""
    result = perform_varcorrcheck(sample_config, sample_dataframe)
    
    # Columns B and E should be dropped (variance)
    # Then between A and C (correlated), one should be dropped
    assert len(result.columns) == 2
    assert 'C' in result.columns  # C should always remain

def test_perform_varcorrcheck_no_checks(sample_dataframe, sample_config):
    """Test when both checks are disabled"""
    sample_config['PROCESSING']['variance_check'] = '0'
    sample_config['PROCESSING']['correlation_check'] = '0'
    
    result = perform_varcorrcheck(sample_config, sample_dataframe)
    
    # No columns should be dropped
    assert list(result.columns) == ['A', 'B', 'C', 'D', 'E']

@patch('reducefeatures.PCA')
def test_perform_pca(mock_pca, sample_dataframe, sample_config, mock_plt):
    """Test PCA functionality"""
    # Setup mock PCA
    mock_pca_instance = MagicMock()
    mock_pca.return_value = mock_pca_instance
    
    # Mock explained variance ratio
    mock_pca_instance.explained_variance_ratio_ = np.array([0.5, 0.3, 0.15, 0.05])
    mock_pca_instance.fit_transform.return_value = np.random.rand(5, 2)  # Mock transformed data
    
    result = perform_pca(sample_config, sample_dataframe)
    
    # Verify PCA was called
    mock_pca.assert_called()
    mock_pca_instance.fit_transform.assert_called()
                
    # Verify result shape (should match mock transformed data)
    assert result.shape == (5, 2)

def test_perform_pca_empty_data(sample_config):
    """Test PCA with empty DataFrame"""
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        perform_pca(sample_config, empty_df)

@patch('reducefeatures.ae_reduce.perform_encoding')
def test_reduce_features_pca(mock_ae, sample_dataframe, sample_config):
    """Test reduce_features with PCA option"""
    sample_config['PROCESSING']['dimensionality_reduction'] = 'pca'
    
    with patch('reducefeatures.perform_pca') as mock_pca:
        mock_pca.return_value = np.random.rand(5, 2)
        result = reduce_features(sample_config, sample_dataframe)
        
        mock_pca.assert_called_once()
        mock_ae.assert_not_called()
        assert isinstance(result, np.ndarray)

@patch('reducefeatures.ae_reduce.perform_encoding')
def test_reduce_features_encoder(mock_ae, sample_dataframe, sample_config):
    """Test reduce_features with encoder option"""
    sample_config['PROCESSING']['dimensionality_reduction'] = 'encoder'
    mock_ae.return_value = np.random.rand(5, 3)
    
    result = reduce_features(sample_config, sample_dataframe)
    
    mock_ae.assert_called_once()
    assert isinstance(result, np.ndarray)

def test_reduce_features_no_reduction(sample_dataframe, sample_config):
    """Test reduce_features with no reduction specified"""
    sample_config['PROCESSING']['variance_check'] = '0'
    sample_config['PROCESSING']['correlation_check'] = '0'
    sample_config['PROCESSING']['dimensionality_reduction'] = 'none'
    
    result = reduce_features(sample_config, sample_dataframe)
        
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_dataframe.shape

def test_perform_varcorrcheck_output_files(sample_dataframe, sample_config, tmpdir):
    """Test that output files are created"""
    sample_config['TEMP']['temp_dir'] = str(tmpdir)
    
    perform_varcorrcheck(sample_config, sample_dataframe)
    
    # Check variance file was created
    assert os.path.exists(os.path.join(str(tmpdir), 'dataset_variance.csv'))
    # Check correlation file was created
    assert os.path.exists(os.path.join(str(tmpdir), 'dataset_correlation_matrix.csv'))

def test_perform_pca_output_files(sample_dataframe, sample_config, tmpdir, mock_plt):
    """Test that PCA output files are created"""
    sample_config['TEMP']['temp_dir'] = str(tmpdir)
    
    # Mock PCA
    with patch('reducefeatures.PCA') as mock_pca:
        mock_pca_instance = MagicMock()
        mock_pca.return_value = mock_pca_instance
        mock_pca_instance.explained_variance_ratio_ = np.array([0.5, 0.3, 0.15, 0.05])
        mock_pca_instance.fit_transform.return_value = np.random.rand(5, 2)
        
        perform_pca(sample_config, sample_dataframe)
    
    # Check plots were saved to the correct directory
    assert os.path.exists(os.path.join(str(tmpdir), 'explained_variance.png'))
    assert os.path.exists(os.path.join(str(tmpdir), 'explained_variance_normalized.png'))
