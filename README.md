# Proteomic aging clock predicts mortality and risk of common age-related diseases in diverse populations

This directory contains the code used for data preparation, analysis, tables, and figure creation for the publication "Proteomic aging clock predicts mortality and risk of common age-related diseases in diverse populations." Published in Nature Medicine (2024). DOI: 10.1038/s41591-024-03164-7.  

This repository was created on Sun July 14 2024.

R and Python code for each stage of our data preparation and analysis are contained in these files. This includes:  
* Importing raw proteomics data
* Training machine learning models for the ProtAge proteomic age clock
* Conducting mortality and incident disease analyses
* Creating figures, tables, and plots


Manifest
--------

The following is a description of the various files and directories found within this project.

|Directory            |Description                                                                                         |
|:--------------------|:---------------------------------------------------------------------------------------------------|
|`dictionaries/`      |Data dictionaries describing the UK Biobank data used in imputation, XWAS, and PheWAS analyses.     |
|`files/`             |Files and coding tables that are called in scripts.                                                 |
|`code/`              |Code used for all analyses and for creating figures/tables.                                         |


Files are in R, R Markdown, Python, and Jupyter Notebook format. R Markdown scripts are not currently written or optimized to be knit and published via R Markdown without additional configuration in the code.

Author
------

Please contact Austin Argentieri (aargentieri@mgh.harvard.edu) with any questions, comments, or concerns.
