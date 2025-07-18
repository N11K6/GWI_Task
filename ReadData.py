#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:06:11 2025

@author: nk
"""
import sys
import pandas as pd
import numpy as np

def assert_format(dataset: pd.DataFrame):
    values_check = dataset.isin([0,1,np.nan]).all()
    if any(values_check==False):
        print('WARNING: Datapoints with incompatible format! \
              Ensure all datapoints are either 0, 1, or NaN!')
        sys.exit(1)
    else:
        print('Dataset format check ok.')
    return 0

def count_missing(dataset: pd.DataFrame):
    n_subjects = dataset.shape[0]
    n_features = dataset.shape[1]
    n_total_datapoints = n_subjects*n_features
    nan_count = dataset.isna().sum().sum()
    print(f'{nan_count} NaN datapoints out of {n_total_datapoints} total datapoints\
          ({np.round(100*nan_count/n_total_datapoints)}%)')
    return 0

def load_dataset(filepath: str) -> pd.DataFrame:
    print('Loading Dataset...')
    dataset = pd.read_excel(filepath)
    assert_format(dataset)
    count_missing(dataset)
    
    return dataset

def main(filepath):
    dataset = load_dataset(filepath)
    return dataset

if __name__ == "__main__":
    filepath = 'dataset.xlsx'
    main(filepath)
    