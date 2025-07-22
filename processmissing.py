#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for processing missing datapoints.

@author: nk
"""
import os
import pandas as pd
import numpy as np
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer
# now we can import normally from sklearn.impute
from sklearn.impute import SimpleImputer
import logging
logger = logging.getLogger(__name__)

from ae_synthesize import synthesize_data

def count_missing(dataset: pd.DataFrame) -> int:
    n_subjects = dataset.shape[0]
    n_features = dataset.shape[1]
    n_total_datapoints = n_subjects*n_features
    nan_count = dataset.isna().sum().sum()
    logger.info(f'{nan_count} NaN datapoints out of {n_total_datapoints} in total.\
          ({np.round(100*nan_count/n_total_datapoints)}%)')
          
    return nan_count

def ignore_missing(dataset: pd.DataFrame) -> pd.DataFrame:
    logger.info('Eliminating all features with missing datapoints.')
    # Ignore missing values
    missing_cols = []
    for col in dataset.columns:
        missing_mask = dataset[col].isna()
        if missing_mask.any():
            missing_cols.append(col)
    dataset = dataset.drop(missing_cols, axis=1)
    
    return dataset

def add_col_missing(dataset: pd.DataFrame) -> pd.DataFrame:
    logger.info('Imputing extra column for missing datapoints.')
    for col in dataset.columns:
        missing_mask = dataset[col].isna()
        if missing_mask.any():
            dataset[col+'_imp'] = np.where(dataset[col].isnull(),1,0)
    dataset.fillna(0,inplace=True)
    return dataset

def impute_missing(dataset: pd.DataFrame) -> pd.DataFrame:
    logger.info('Imputing most frequent value for missing datapoints.')
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    dataset_imputed = pd.DataFrame(
        simple_imputer.fit_transform(dataset),
        columns=dataset.columns,
        index=dataset.index
        )
    return dataset_imputed

def handle_missing(config, dataset: pd.DataFrame) -> pd.DataFrame:
    count_missing(dataset)
    missing_strategy = config['PROCESSING']['missing_strategy']
    logger.info(f'Missing data handled through strategy: {missing_strategy}')
    if missing_strategy == 'ignore':
        dataset = ignore_missing(dataset)
    elif missing_strategy == 'add':
        dataset = add_col_missing(dataset)
    elif missing_strategy == 'impute':
        dataset = impute_missing(dataset)
    elif missing_strategy == 'synthesize':
        dataset = synthesize_data(config, dataset)
    else:
        raise ValueError('missing_strategy can only be: ignore, impute, or synthesize')
    # Save the dataset without missing datapoints
    output_dir = config['TEMP']['temp_dir']
    dataset.to_csv(os.path.join(output_dir, 'dataset_nomissing.csv'),index=False)
    
    return dataset
