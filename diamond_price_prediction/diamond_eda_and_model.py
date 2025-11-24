# Databricks notebook source
# DBTITLE 1,imports
import pandas as pd

from pyspark.sql.functions import col, when, count, abs, log, exp
from pyspark.ml.feature import SQLTransformer, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------


# Define the path to the built-in dataset
file_path = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv"

# COMMAND ----------

df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(file_path)

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.summary())

# COMMAND ----------

# Check for null values, but this dataset is so known, so it should not be a problem
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# COMMAND ----------

# Check for dimensional errors, in the x, y and z columns
bad_diamonds = df.filter((col("x") == 0) | (col("y") == 0) | (col("z") == 0))
display(bad_diamonds)

# COMMAND ----------

# There are 20 rows with 0 in either x, y or z, so we need to remove those, it is few in relation to the larger dataset
df_clean = df.filter((col("x") > 0) & (col("y") > 0) & (col("z") > 0) )

assert df.count() == df_clean.count() + bad_diamonds.count(), "Cleaning procedure did not remove the rows with 0 in x, y or z"

# COMMAND ----------

# The size and carat are likely correlated, so lets estimate the correlation
corr_x = df_clean.stat.corr("carat", "x")
corr_y = df_clean.stat.corr("carat", "y")
corr_z = df_clean.stat.corr("carat", "z")

print(f"Correlation Carat vs X: {corr_x}")
print(f"Correlation Carat vs Y: {corr_y}")
print(f"Correlation Carat vs Z: {corr_z}")

# COMMAND ----------

# We see that the correlation is high > 0.95, so we remove those columns and only keep the carat
df_final = df_clean.drop("x", "y", "z")
print(df_final.columns)

# COMMAND ----------

categoical_columns = ['cut', 'color', 'clarity']
number_columsn = ['carat', 'depth', 'table']
prediction_columns = ['price']

# COMMAND ----------

# We want ordinal encoding here, so that the relationshjip between the classes are somewhat kept, with the higher the better
ordinal_sql_query = """
SELECT *,
    -- Cut Score Logic
    CASE 
        WHEN cut = 'Fair' THEN 1 
        WHEN cut = 'Good' THEN 2 
        WHEN cut = 'Very Good' THEN 3 
        WHEN cut = 'Premium' THEN 4 
        WHEN cut = 'Ideal' THEN 5 
        ELSE 0 
    END as cut_score,

    -- Color Score Logic
    CASE 
        WHEN color = 'J' THEN 1 
        WHEN color = 'I' THEN 2 
        WHEN color = 'H' THEN 3 
        WHEN color = 'G' THEN 4 
        WHEN color = 'F' THEN 5 
        WHEN color = 'E' THEN 6 
        WHEN color = 'D' THEN 7 
        ELSE 0 
    END as color_score,

    -- Clarity Score Logic
    CASE 
        WHEN clarity = 'I1' THEN 1 
        WHEN clarity = 'SI2' THEN 2 
        WHEN clarity = 'SI1' THEN 3 
        WHEN clarity = 'VS2' THEN 4 
        WHEN clarity = 'VS1' THEN 5 
        WHEN clarity = 'VVS2' THEN 6 
        WHEN clarity = 'VVS1' THEN 7 
        WHEN clarity = 'IF' THEN 8 
        ELSE 0 
    END as clarity_score

FROM __THIS__
"""
ordinal_stage = SQLTransformer(statement=ordinal_sql_query)

# COMMAND ----------

# Define features
feature_columns = ['carat', 'depth', 'table', 'cut_score', 'color_score', 'clarity_score']
assembeler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# COMMAND ----------

lr = LinearRegression(featuresCol="features", labelCol="price")

pipeline = Pipeline(stages=[ordinal_stage, assembeler, lr])

train_data, test_data = df_final.randomSplit([0.75, 0.25], seed=42)

# Train the model
model = pipeline.fit(train_data)
# Make predictions
predictions = model.transform(test_data)

# COMMAND ----------

def evaluate_model(predictions, model_name="Generic Model", label_col="price"):
    """
    Calculates and prints R2, RMSE, and MAE for a Spark Regression Model.
    """
    eval_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")
    eval_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    eval_mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")

    r2 = eval_r2.evaluate(predictions)
    rmse = eval_rmse.evaluate(predictions)
    mae = eval_mae.evaluate(predictions)
    
    print(f"--- Evaluation: {model_name} ---")
    print(f"R² Score:  {r2:.4f}")
    print(f"RMSE:      ${rmse:,.2f}")
    print(f"MAE:       ${mae:,.2f}")
    print("-" * 35)
    return {"r2": r2, "rmse": rmse, "mae": mae}

# COMMAND ----------

eval = evaluate_model(predictions, model_name="LinearRegression")

# COMMAND ----------

# We add an 'error' column and an 'absolute_error' column
analysis_df = predictions.withColumn("error", col("price") - col("prediction")) \
                         .withColumn("abs_error", abs(col("price") - col("prediction")))
# Sort by worst errors to see what went wrong
display(analysis_df.select("price", "prediction", "error", "carat", "cut").orderBy(col("abs_error").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation: Linear Regression: Baseline
# MAGIC --- Evaluation: LinearRegression ---
# MAGIC * R² Score:  0.9001
# MAGIC * RMSE:      $1,246.08
# MAGIC * MAE:       $859.70
# MAGIC
# MAGIC Analysis:
# MAGIC While the linear model captures the general trend, the high RMSE suggests that the relationship between diamond features and price is non-linear. Diamond prices tend to jump exponentially (e.g., a 2.0-carat diamond is much more than 2x the price of a 1.0-carat diamond).
# MAGIC
# MAGIC Next Step:
# MAGIC Train a Random Forest Regressor. Tree-based models are better suited to capture non-linear patterns and threshold-based pricing jumps without requiring complex feature transformation.
# MAGIC

# COMMAND ----------

rf = RandomForestRegressor(featuresCol="features", labelCol="price", numTrees=15, maxDepth=20)
pipeline_rf = Pipeline(stages=[ordinal_stage, assembeler, rf])
model_rf = pipeline_rf.fit(train_data)
predictions_rf = model_rf.transform(test_data)
eval_rf = evaluate_model(predictions_rf, model_name="RandomForestRegressor")

# COMMAND ----------

# We create dummy data for an ideal diamond, it should be above 20k
data = [(3.0, "Ideal", "D", "IF", 62.0, 55.0)]
schema = ["carat", "cut", "color", "clarity", "depth", "table"]
my_diamond = spark.createDataFrame(data, schema)
my_prediction = model_rf.transform(my_diamond)

display(my_prediction.select("carat", "cut", "color", "clarity", "prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation: Random Forest Regressor (Tuned)
# MAGIC --- Evaluation: RandomForestRegressor ---
# MAGIC * R² Score:  0.9678
# MAGIC * RMSE:      $707.13
# MAGIC * MAE:       $367.18
# MAGIC
# MAGIC Analysis:
# MAGIC The Random Forest model dramatically outperformed the linear baseline by capturing the non-linear thresholds in the data (e.g., specific quality combinations). The average error dropped from ~$860 to ~$367. 
# MAGIC
# MAGIC However, the model failed to extrapolate the price of a theoretical "Perfect Diamond" (predicting $16,756 instead of $50,000+). This confirms that tree-based models are bounded by the maximum value in their training set (~$18,823) and cannot predict outside that range.
# MAGIC
# MAGIC Next Step:
# MAGIC To squeeze the final percentage points of performance, we could try Gradient Boosted Trees (GBT), which correct errors sequentially, suggested in Kaggle competetions

# COMMAND ----------

gbt = GBTRegressor(featuresCol="features", labelCol="price", maxIter=100, maxDepth=8)
pipeline_gbt = Pipeline(stages=[ordinal_stage, assembeler, gbt])

print("Training Gradient Boosted Trees...")
model_gbt = pipeline_gbt.fit(train_data)
predictions_gbt = model_gbt.transform(test_data)

# Evaluate
evaluate_model(predictions_gbt, model_name="GBT Regressor")

# COMMAND ----------

my_prediction = model_gbt.transform(my_diamond)
display(my_prediction.select("carat", "cut", "color", "clarity", "prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Analysis Gradient Boosted Trees
# MAGIC We see that the performance is not particularly better than the RF
# MAGIC
# MAGIC Training Gradient Boosted Trees...
# MAGIC --- Evaluation: GBT Regressor ---
# MAGIC * R² Score:  0.9676
# MAGIC * RMSE:      $710.01
# MAGIC * MAE:       $332.86
# MAGIC
# MAGIC While GBT offered a lower MAE, it did not significantly improve the $R^2$ or RMSE. The model appears to be more sensitive to outliers, punishing the RMSE score despite better average performance.
# MAGIC
# MAGIC Since algorithm tuning has hit a plateau (~0.97 $R^2$), we will see if we can do some more Feature Engineering.
# MAGIC
# MAGIC Diamond prices follow an exponential curve, but regression models prefer normal distributions.
# MAGIC By training the model on the Logarithm of the Price ($\log(price)$), we can compress the massive price range into a linear scale, potentially helping the Random Forest capture the trend more accurately.

# COMMAND ----------

# MAGIC %md
# MAGIC # Attempt to log(price)

# COMMAND ----------

df_log = df_final.withColumn("log_price", log(col("price")))
train_log, test_log = df_log.randomSplit([0.8, 0.2], seed=42)
rf_log = RandomForestRegressor(featuresCol="features", labelCol="log_price", numTrees=15, maxDepth=20)
pipeline_log = Pipeline(stages=[ordinal_stage, assembeler, rf_log])

print("Training Log-Transformed Model...")
model_log = pipeline_log.fit(train_log)
raw_predictions = model_log.transform(test_log)

predictions_final = raw_predictions.withColumn("prediction_dollars", exp(col("prediction")))
predictions_for_eval = predictions_final.drop("prediction").withColumnRenamed("prediction_dollars", "prediction")

evaluate_model(predictions_for_eval, model_name="Log-Transformed Random Forest")


# COMMAND ----------

# MAGIC %md
# MAGIC # Analysis Log-Transformation Experiment
# MAGIC
# MAGIC --- Evaluation: Log-Transformed Random Forest ---
# MAGIC * R² Score:  0.9584
# MAGIC * RMSE:      $807.36
# MAGIC * MAE:       $387.05

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rf_stage = model_rf.stages[-1] # The last stage is the model
assembler_stage = model_rf.stages[1] # The 2nd stage is the VectorAssembler

# 2. Map Numbers back to Names
# The model only knows "Feature 0", "Feature 1". We need names.
feature_names = assembler_stage.getInputCols()
importances = rf_stage.featureImportances.toArray()

# 3. Create a Pandas DataFrame for Plotting
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 4. Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette="viridis")
plt.title('What Drives Diamond Prices? (Model Interpretation)')
plt.xlabel('Relative Importance (0.0 - 1.0)')
plt.show()
