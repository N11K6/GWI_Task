#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for running the project via Flask API. 
Required inputs are a dataset (excel or csv)
and a ini file with the appropriate format and information.

@author: nk
"""
from flask import Flask, request, send_file, render_template
import pandas as pd
import configparser
import os
import tempfile
import shutil
from zipfile import ZipFile
from io import BytesIO
import uuid
import logging
logger = logging.getLogger(__name__)

import processmissing
import reducefeatures
import clustering

app = Flask(__name__)
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

def read_data_file(file_path):
    """Read either Excel or CSV file based on extension"""
    if file_path.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    elif file_path.lower().endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format")

def main_process(config, dataset):
    
    output_dir = config['TEMP']['temp_dir']
    
    logging.basicConfig(filename=os.path.join(output_dir, 'applog.log'), level=logging.INFO)    
    logger.info('Started')
    
    dataset = processmissing.handle_missing(config, dataset)
    data_reduced = reducefeatures.reduce_features(config, dataset)
    data_clustered, metrics_dict = clustering.perform_clustering(config, data_reduced)
    
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
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'data_file' not in request.files or 'ini_file' not in request.files:
            return "Both data file and INI file are required", 400
            
        data_file = request.files['data_file']
        ini_file = request.files['ini_file']
        
        if data_file.filename == '' or ini_file.filename == '':
            return "No selected files", 400
            
        # Get the filenames as strings for extension checking
        data_filename = data_file.filename.lower()
        ini_filename = ini_file.filename.lower()
            
        if (not data_filename.endswith(('.xls', '.xlsx', '.csv'))) or \
           (not ini_filename.endswith('.ini')):
            return "Invalid file types. Data file must be Excel or CSV, config must be INI", 400      
        
        try:
            # Create a unique temporary directory for this processing run
            temp_dir = os.path.join(tempfile.gettempdir(), f'process_{uuid.uuid4().hex}')
            os.makedirs(temp_dir, exist_ok=True)

            # Read input files
            df = read_data_file(data_filename)
            config = configparser.ConfigParser()
            config.read_string(ini_file.read().decode('utf-8'))

            # Store temporary directory path to config
            config.add_section('TEMP')
            config.set('TEMP','temp_dir', temp_dir)
            
            # Call the main process (outputs will be saved to temp_dir)
            main_process(config, df)
            
            # Create zip file with all contents of temp_dir
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, 'w') as zip_file:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zip_file.write(file_path, arcname)
            
            zip_buffer.seek(0)
            
            # Clean up temporary files
            shutil.rmtree(temp_dir)
            
            return send_file(
                zip_buffer,
                as_attachment=True,
                download_name='processed_results.zip',
                mimetype='application/zip'
            )
            
        except Exception as e:
            # Clean up temp directory if it exists
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            return f"Error processing files: {str(e)}", 500
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)