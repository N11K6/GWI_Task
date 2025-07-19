#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:06:11 2025

@author: nk
"""
import configparser
import pandas as pd
import numpy as np

def read_config():
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.ini')
    return config

def assert_format(dataset: pd.DataFrame) -> pd.DataFrame:
    values_check = dataset.isin([0,1,np.nan]).all()
    if any(values_check==False):
        print('WARNING: Datapoints with incompatible format! \n Replacing all values not in [0, 1] with NaN!')
        dataset = dataset.where(dataset.isin([0, 1, np.nan]))
    else:
        print('Datapoint format check ok.')
    return dataset
    
def load_check_dataset(config: configparser.ConfigParser) -> pd.DataFrame:
    filepath = config['DATASOURCE']['filepath']
    print(f'Loading Dataset from {filepath}...')
    dataset = pd.read_excel(filepath)
    dataset = assert_format(dataset)
    return dataset

if __name__ == "__main__":
    config = read_config()    
    dataset = load_check_dataset(config)
    