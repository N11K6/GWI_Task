# GWI_Task
Technical Task for Position at GWI

## Contents

### Code:
* appMain.py: The main module to run, which uses Flask API. 
* processmissing.py: Module containing functions for handling missing data.
* ae_synthesize.py: Sub-module called by processmissing.py for synthesizing data using an Autoencoder
* reducefeatures.py: Module containing functions for feature reduction.
* ae_reduce.py: Sub-module called by reducefeatures,.py for dimensionality reduction using an Autoencoder.
* clustering.py: Module containing functions for clustering.
* Testing modules: Unit tests for each module (test_processmissing.py, test_reducefeatures.py, test_clustering.py), and a test for the API implementation (test_appMain.py).
* ***I have also added debugMain.py - a simple script to run via command line in the same directory, for a simple run of the pipeline. It will generate a temp_dir folder storing the outputs. This is not part of the deployment, just to help the developer with debugging etc.***

### Additional Files:
* dataset.xlsx: The sample dataset that was provided.
* config.ini: Configuration file with the parameters for the pipeline.
* templates/upload.html: Html file for API.
* pytest.ini: Configuration for testing.
* requrements.txt: Required libraries for running the pipeline.
* README.md: This file.

## Usage:
Run "python3 appMain.py". By default runs localy on http://127.0.0.1:5000. 
A dataset in excel or csv format, and an appropriate config ini file must be uploaded for the pipeline to run.
Once the pipeline runs, it will provide a zip file for download with all the outputs.
