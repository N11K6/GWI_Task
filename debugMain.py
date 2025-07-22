#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is not part of the deployed pipeline, but is useful for debugging.
It calls the modules in the same order, performing data processing and clustering.
Results are saved in a temp_dir directory.

Required: A valid config.ini in the directory, pointing to a dataset location ([DATASOURCE][filepath])

@author: nk
"""
import os
import shutil
import configparser
import pandas as pd
import logging
logger = logging.getLogger(__name__)

import processmissing
import reducefeatures
import clustering

def read_data_file(file_path: str):
    """Read either Excel or CSV file based on extension"""
    if file_path.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    elif file_path.lower().endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format")

def main():
    # Process configuration
    config = configparser.ConfigParser()
    config.read('config.ini')
    config.add_section('TEMP')
    output_dir = 'temp_dir'
    config.set('TEMP','temp_dir', output_dir)
    # Remove directory if it exists (along with all contents)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    # Create fresh empty directory
    os.makedirs(output_dir)
    # Clear existing handlers (if any)
    logging.root.handlers = []
    
    # Set up fresh logging
    logging.basicConfig(
    filename=os.path.join(output_dir, 'applog.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
        )
    logger.info('Started')
    
    # Read dataset
    filepath = config['DATASOURCE']['filepath']
    dataset = read_data_file(filepath)
    
    # Start pipeline
    dataset = processmissing.handle_missing(config, dataset)
    data_reduced = reducefeatures.reduce_features(config, dataset)
    data_clustered, metrics_dict = clustering.perform_clustering(config, data_reduced)
    
    # Prepare results for output
    data_clustering_results = dataset.copy()
    data_clustering_results['cluster'] = data_clustered
    metrics_df = pd.DataFrame(metrics_dict.items())
    
    # Save CSV: Reduced data for clustering
    pd.DataFrame(data_reduced).to_csv(os.path.join(output_dir, 'dataset_reduced.csv'),index=False)
    
    # Save CSV: Data clustering results
    data_clustering_results.to_csv(os.path.join(output_dir, 'data_clustering_results.csv'), index=False)
    
    # Save CSV: Clustering evaluation metrics
    metrics_df.to_csv(os.path.join(output_dir, 'clustering_evaluation_metrics.csv'), index=False)
    
    logger.info('Finished')

if __name__ == "__main__":
    main()
