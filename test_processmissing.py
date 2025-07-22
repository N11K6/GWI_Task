#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit testing for processmissing

@author: nk
"""

import pytest
import pandas as pd
import numpy as np
from processmissing import (
    count_missing,
    ignore_missing,
    add_col_missing,
    impute_missing,
    handle_missing
)
from unittest.mock import patch
import logging

# Setup logging for testing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_dataframe_with_nans():
    """Fixture providing a sample DataFrame with NaN values"""
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [9, 10, 11, 13],
        'D': [np.nan, np.nan, 15, np.nan]
    })

@pytest.fixture
def sample_config():
    """Fixture providing a sample config dictionary"""
    return {
        'PROCESSING': {
            'missing_strategy': 'impute',
            'imputation_strategy': 'mean'
        },
        'TEMP': {
            'temp_dir': '/tmp'
        }
    }

def test_count_missing(sample_dataframe_with_nans):
    """Test counting missing values"""
    result = count_missing(sample_dataframe_with_nans)        
    assert result == 6  # Total NaN values in the sample dataframe

def test_ignore_missing(sample_dataframe_with_nans):
    """Test ignoring columns with missing values"""
    result = ignore_missing(sample_dataframe_with_nans)
    
    # Only column C should remain (no NaN values)
    assert list(result.columns) == ['C']
    assert result.shape == (4, 1)

def test_add_col_missing(sample_dataframe_with_nans):
    """Test numerical imputation with a specific value"""
    result = add_col_missing(sample_dataframe_with_nans)
    
    # Check all NaN values were replaced with -1
    assert result.isna().sum().sum() == 0
    assert (result.loc[2, 'A'] == 0)
    assert (result.loc[2, 'A_imp'] == 1)
    assert (result.loc[1, 'B'] == 0)
    assert (result.loc[1, 'B_imp'] == 1)
    assert (result.loc[3, 'D'] == 0)
    assert (result.loc[3, 'D_imp'] == 1)

def test_impute_missing(sample_dataframe_with_nans):
    """Test SimpleImputer with most_frequent strategy"""
    df = sample_dataframe_with_nans.copy()
    df.loc[0, 'A'] = 2  # Now 2 appears twice in column A
    result = impute_missing(df)
    
    assert result.isna().sum().sum() == 0
    # Most frequent in A is 2
    assert result.loc[2, 'A'] == 2
    # Most frequent in B is 5 or 8 (both appear once)
    assert result.loc[1, 'B'] in [5, 8]

def test_handle_missing_ignore(sample_dataframe_with_nans, sample_config):
    """Test handle_missing with 'ignore' strategy"""
    sample_config['PROCESSING']['missing_strategy'] = 'ignore'
    result = handle_missing(sample_config, sample_dataframe_with_nans)
    
    assert list(result.columns) == ['C']  # Only column without NaNs

def test_handle_missing_impute(sample_dataframe_with_nans, sample_config):
    """Test handle_missing with 'impute' strategy"""
    sample_config['PROCESSING']['missing_strategy'] = 'impute'
    result = handle_missing(sample_config, sample_dataframe_with_nans)
    
    assert result.isna().sum().sum() == 0

@patch('processmissing.synthesize_data')
def test_handle_missing_synthesize(mock_synthesize, sample_dataframe_with_nans, sample_config):
    """Test handle_missing with 'synthesize' strategy"""
    sample_config['PROCESSING']['missing_strategy'] = 'synthesize'
    mock_synthesize.return_value = sample_dataframe_with_nans.fillna(0)
    result = handle_missing(sample_config, sample_dataframe_with_nans)
    
    mock_synthesize.assert_called_once()
    assert result.isna().sum().sum() == 0

def test_handle_missing_invalid_strategy(sample_dataframe_with_nans, sample_config):
    """Test handle_missing with invalid strategy"""
    sample_config['PROCESSING']['missing_strategy'] = 'invalid'
    
    with pytest.raises(ValueError):
        handle_missing(sample_config, sample_dataframe_with_nans)
