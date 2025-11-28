# Classification Example in Databricks

This project provides an end to end classification example built in Databricks using the Adult Census dataset. The goal is to predict whether an individual earns more or less than 50k based on demographic and employment features.

## Overview

- Loading and preparing the Adult Census dataset  
- Handling missing values  
- Training and comparing several machine learning models  
- Evaluating the best model on a holdout test set  
- Exploring feature importance to understand key drivers in the predictions  

## Constraints and Considerations

The workflow was developed in the Databricks Community Edition. Due to resource limits, some high dimensional features such as *native country* were removed. The same limitations affected the amount of hyperparameter tuning and the number of model variants that could be tested.

## Results

The final model performs reasonably well given the restricted environment. Several steps were skipped or simplified because of resource constraints, for example:

- Retaining all original features  
- Exploring multiple imputation strategies   
- Running broader hyperparameter searches  
- Trying more complex models  

## Purpose

This example illustrates a simple but complete classification pipeline in Databricks, from data preparation to model evaluation and interpretation.
