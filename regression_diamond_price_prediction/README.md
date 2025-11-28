# Diamond Price Prediction in Databricks

This project demonstrates a full regression workflow in Databricks using the well known **Diamonds** dataset. The goal is to predict the price of a diamond from a set of physical and quality related features. The notebook walks through data cleaning, feature engineering, model training, evaluation, and model interpretation.

## Dataset

The project uses the built in Databricks copy of the `ggplot2::diamonds` dataset: /databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv


It contains attributes such as carat, cut, color, clarity, depth, and size dimensions, together with the final market price.

## Workflow

### 1. Loading and Inspecting the Data
The notebook begins by loading the CSV file into a Spark DataFrame. Summary statistics and basic null checks ensure the dataset is ready for processing.

### 2. Data Cleaning
A small number of rows contain invalid geometric measurements where `x`, `y`, or `z` equals zero. These rows are removed. A sanity check confirms that the cleaning step removes the correct number of records.

### 3. Exploratory Analysis
The dimensional columns `x`, `y`, and `z` are highly correlated with carat (> 0.95). Since they contain redundant information, they are removed to keep the feature space compact.

### 4. Ordinal Encoding of Categorical Features
Cut, color, and clarity have an inherent order. An `SQLTransformer` is used to apply ordinal encoding directly inside a Spark pipeline. For example:

- Cut: Fair → Ideal  
- Color: J → D  
- Clarity: I1 → IF

This preserves the rank information needed for regression models.

### 5. Feature Assembly
All numerical and encoded categorical variables are merged into a single feature vector using a `VectorAssembler`.

### 6. Model Training
Several regression models are trained and compared:

#### **Linear Regression**
Used as a baseline.  
Performance:  
- R² ≈ 0.90  
- RMSE ≈ $1,246  
- MAE ≈ $860  

Linear models fail to capture the non-linear pricing behaviour of diamonds.

#### **Random Forest Regressor**
A tree based model that handles non-linearities well.  
Performance improved significantly:  
- R² ≈ 0.97  
- RMSE ≈ $707  
- MAE ≈ $367  

A custom inspection step also evaluates very expensive theoretical diamonds and highlights the model’s inability to extrapolate above the observed training values.

#### **Gradient Boosted Trees**
Performance is similar to Random Forest.  
- Slightly lower MAE  
- Similar R² and RMSE  
- More sensitive to outliers  

### 7. Optional Log Transformation
Since diamond prices follow an exponential curve, a log transform is tested. The model is trained on `log(price)` and predictions are exponentiated afterwards. This transformation did not outperform the standard Random Forest configuration.

### 8. Model Interpretation
Feature importance from the Random Forest model is extracted and visualised in a Python plotting cell. This highlights which attributes contribute most to predicting diamond prices.

Typical ranking:
1. Carat  
2. Clarity score  
3. Color score  
4. Cut score  
5. Depth and table  

## Results Summary

- Tree based models clearly outperform linear regression.  
- Random Forest provides the best balance of R², RMSE, and MAE in this environment.  
- Gradient Boosted Trees offer only marginal improvements.  
- Log transforming the price did not improve performance in this setup.  
- The final model reaches **~0.97 R²**, consistent with expected results on this dataset.

## Purpose

The notebook serves as a structured example of:

- Data cleaning in Spark  
- Feature engineering with SQLTransformer and VectorAssembler  
- Training multiple regression models in a Databricks pipeline  
- Evaluating models with standard Spark ML evaluators  
- Inspecting errors and feature importance  
- Using Databricks to explore non-linear modelling behaviour  

It is designed as an accessible introduction to regression modelling with PySpark in a Databricks environment.

