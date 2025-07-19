#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 12:36:25 2025

@author: nk
"""
import pandas as pd
import numpy as np
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer
# now we can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

import dataload
from ae_synthesize import synthesize_data

def count_missing(dataset: pd.DataFrame) -> int:
    n_subjects = dataset.shape[0]
    n_features = dataset.shape[1]
    n_total_datapoints = n_subjects*n_features
    nan_count = dataset.isna().sum().sum()
    print(f'{nan_count} NaN datapoints out of {n_total_datapoints} in total.\
          ({np.round(100*nan_count/n_total_datapoints)}%)')
    return nan_count

def ignore_missing(dataset):
    # Ignore missing values
    missing_cols = []
    for col in dataset.columns:
        missing_mask = dataset[col].isna()
        if missing_mask.any():
            missing_cols.append(col)
    dataset = dataset.drop(missing_cols, axis=1)
    
    return dataset

def perform_imputation_num(dataset: pd.DataFrame, fill_value: int) -> pd.DataFrame:
    dataset_imputed = dataset.fillna(fill_value)
    return dataset_imputed

def perform_SimpleImp(dataset: pd.DataFrame, imputation_strategy) -> pd.DataFrame:
    print(f'Imputing {imputation_strategy} value for missing datapoints.')
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy)
    dataset_imputed = pd.DataFrame(
        simple_imputer.fit_transform(dataset),
        columns=dataset.columns,
        index=dataset.index
        )
    return dataset_imputed

def perform_MICE(dataset: pd.DataFrame) -> pd.DataFrame:
    print('Imputation strategy chosen: MICE. Performing Iterative Imputation...')
    iterative_imputer = IterativeImputer(random_state=42, max_iter=10)
    dataset_imputed = pd.DataFrame(
        iterative_imputer.fit_transform(dataset),
        columns=dataset.columns,
        index=dataset.index
        )
    return dataset_imputed

def impute_missing(config, dataset):
    imputation_strategy = config['PROCESSING']['imputation_strategy'].lower()
    if imputation_strategy == 'other':
        dataset = perform_imputation_num(dataset, -1)
    else:
        dataset = perform_SimpleImp(dataset, imputation_strategy)
    return dataset

def handle_missing(config, dataset):
    missing_strategy = config['PROCESSING']['missing_strategy']
    print(f'Missing data handled through strategy: {missing_strategy}')
    if missing_strategy == 'ignore':
        dataset = ignore_missing(dataset)
    elif missing_strategy == 'impute':
        dataset = impute_missing(config,dataset)
    elif missing_strategy == 'synthesize':
        dataset = synthesize_data(config, dataset)
    return dataset

if __name__ == "__main__":
    config = dataload.read_config()
    dataset = dataload.load_check_dataset(config)
    _ = count_missing(dataset)
    dataset = handle_missing(config, dataset)
