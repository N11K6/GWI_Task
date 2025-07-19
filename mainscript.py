#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 11:02:35 2025

@author: nk
"""
import dataload
import processmissing
import reducefeatures
import clustering

if __name__ == "__main__":
    config = dataload.read_config()    
    dataset = dataload.load_check_dataset(config)
    dataset = processmissing.handle_missing(config, dataset)
    data_reduced = reducefeatures.reduce_features(config, dataset)
    data_clustered = clustering.perform_clustering(config, dataset, data_reduced)
