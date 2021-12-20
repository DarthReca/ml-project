# Machine Learning Project

## Description

This project was built during the course of _Machine Learning and Pattern Recognition_ at Polytechnic University of Turin. The purpose was to build models and tools to analize the dataset **HTRU2** (_R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656_)

## Modules

- cross_validation
- data_loading
- data_plotting
- data_result_analysis
- dimensionality_reduction
- models
- preprocess

### Cross Validation

Utilities function to do cross validation

### data_loading

Utilities function to load training data and test data

### data_plotting

Functions to plot scatter, histograms and heat-maps

### data_result_analysis

Functions to compute:

- scalar metrics (precision, F-beta-score, Matthews Correlation Coefficient)
- confusion matrix
- minimum DCF and DCF
- ROC, DET
- Bayes error plot

### dimensionality_reduction

Function to compute LDA and PCA

### models

Classes for different ML models

### preprocess

Classes to gaussianize and preprocess data
