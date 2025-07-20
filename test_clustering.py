#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit testing for clustering

@author: nk
"""

import pytest
import numpy as np
from clustering import (
    perform_HDBSCAN,
    plot2d,
    plot3d,
    perform_clustering
)
from unittest.mock import patch, MagicMock, call
import logging
import os
from sklearn.exceptions import NotFittedError

# Setup logging for testing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_data_2d():
    """Fixture providing 2D sample data"""
    return np.array([
        [1.0, 2.0],
        [1.1, 2.1],
        [5.0, 6.0],
        [5.1, 6.1],
        [10.0, 11.0]
    ])

@pytest.fixture
def sample_data_3d():
    """Fixture providing 3D sample data"""
    return np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [5.0, 6.0, 7.0],
        [5.1, 6.1, 7.1],
        [10.0, 11.0, 12.0]
    ])

@pytest.fixture
def sample_clusters():
    """Fixture providing sample cluster labels"""
    return np.array([0, 0, 1, 1, 2])

@pytest.fixture
def sample_config():
    """Fixture providing a sample config dictionary"""
    return {
        'CLUSTERING': {
            'min_cluster_size': '2'
        },
        'TEMP': {
            'temp_dir': '/tmp'
        }
    }

@patch('clustering.HDBSCAN')
def test_perform_HDBSCAN(mock_hdbscan, sample_config, sample_data_2d):
    """Test HDBSCAN clustering"""
    # Setup mock HDBSCAN
    mock_hdbscan_instance = MagicMock()
    mock_hdbscan.return_value = mock_hdbscan_instance
    mock_hdbscan_instance.fit_predict.return_value = np.array([0, 0, 1, 1, 2])
    
    result = perform_HDBSCAN(sample_config, sample_data_2d)
    
    # Verify HDBSCAN was called with correct parameters
    mock_hdbscan.assert_called_once_with(min_cluster_size=2)
    mock_hdbscan_instance.fit_predict.assert_called_once_with(sample_data_2d)
    
    # Verify result
    assert isinstance(result, np.ndarray)
    assert len(result) == len(sample_data_2d)

@patch('clustering.perform_HDBSCAN')
@patch('clustering.plot2d')
@patch('clustering.plot3d')
@patch('clustering.silhouette_score')
@patch('clustering.davies_bouldin_score')
@patch('clustering.calinski_harabasz_score')
def test_perform_clustering_2d(mock_ch, mock_db, mock_sil, mock_plot3d, mock_plot2d, 
                             mock_hdbscan, sample_config, sample_data_2d):
    """Test perform_clustering with 2D data"""
    # Setup mocks
    mock_hdbscan.return_value = np.array([0, 0, 1, 1, 2])
    mock_sil.return_value = 0.75
    mock_db.return_value = 0.5
    mock_ch.return_value = 100.0
    
    clusters, metrics = perform_clustering(sample_config, sample_data_2d)
    
    # Verify HDBSCAN was called
    mock_hdbscan.assert_called_once_with(sample_config, sample_data_2d)
    
    # Verify 2D plot was called (not 3D)
    mock_plot2d.assert_called_once()
    mock_plot3d.assert_not_called()
    
    # Verify metrics were calculated
    mock_sil.assert_called_once()
    mock_db.assert_called_once()
    mock_ch.assert_called_once()
    
    # Verify return values
    assert isinstance(clusters, np.ndarray)
    assert isinstance(metrics, dict)
    assert metrics['sil_score'] == 0.75
    assert metrics['db_score'] == 0.5
    assert metrics['ch_score'] == 100.0

@patch('clustering.perform_HDBSCAN')
@patch('clustering.plot2d')
@patch('clustering.plot3d')
def test_perform_clustering_3d(mock_plot3d, mock_plot2d, mock_hdbscan, 
                             sample_config, sample_data_3d):
    """Test perform_clustering with 3D data"""
    # Setup mocks
    mock_hdbscan.return_value = np.array([0, 0, 1, 1, 2])
    
    clusters, _ = perform_clustering(sample_config, sample_data_3d)
    
    # Verify 3D plot was called (not 2D)
    mock_plot3d.assert_called_once()
    mock_plot2d.assert_not_called()

def test_perform_clustering_invalid_data(sample_config):
    """Test perform_clustering with invalid data"""
    # 1D data (invalid)
    data_1d = np.array([1, 2, 3, 4, 5])
    
    with pytest.raises(ValueError):
        perform_clustering(sample_config, data_1d)

@patch('clustering.HDBSCAN')
def test_perform_HDBSCAN_failure(mock_hdbscan, sample_config, sample_data_2d):
    """Test HDBSCAN clustering failure"""
    # Setup mock to raise exception
    mock_hdbscan_instance = MagicMock()
    mock_hdbscan.return_value = mock_hdbscan_instance
    mock_hdbscan_instance.fit_predict.side_effect = Exception("Clustering failed")
    
    with pytest.raises(Exception, match="Clustering failed"):
        perform_HDBSCAN(sample_config, sample_data_2d)

@patch('clustering.perform_HDBSCAN')
def test_perform_clustering_empty_data(mock_hdbscan, sample_config):
    """Test perform_clustering with empty data"""
    empty_data = np.array([])
    
    with pytest.raises(ValueError):
        perform_clustering(sample_config, empty_data)
