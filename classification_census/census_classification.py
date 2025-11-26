# Databricks notebook source
from pyspark.sql.functions import col, trim, when, lit, coalesce, row_number, count
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer, StandardScaler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# COMMAND ----------

dataset_path = "/databricks-datasets/adult/adult.data"

# COMMAND ----------

df_raw = spark.read.format("csv").option("inferSchema", "true").load(dataset_path)

# COMMAND ----------

display(df_raw)
display(df_raw.summary())

# COMMAND ----------

# Column name are missing, found at https://www.kaggle.com/datasets/mlbysoham/adult-dataset
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", 
           "hours_per_week", "native_country", "income"]
for i, name in enumerate(columns):
    df_raw = df_raw.withColumnRenamed(f"_c{i}", name)

# COMMAND ----------

display(df_raw)

# COMMAND ----------

# Verify the income class to get a sense of the distribution
above_50k  = df_raw.filter(col("income") == ">50K").count()
print(above_50k)

# COMMAND ----------

# Get 0, with that key, lets check for distinct classes
df_raw.select("income").distinct().show()

# COMMAND ----------

# We must trim the whitespace for string columns to work properly
# Loop through all columns; if it's a string, trim the whitespace
for c_name, c_type in df_raw.dtypes:
    if c_type == "string":
        df_raw = df_raw.withColumn(c_name, trim(col(c_name)))

above_50k  = df_raw.filter(col("income") == ">50K").count()
print(above_50k)
below_50k = df_raw.filter(col("income") == "<=50K").count()
print(below_50k)

# COMMAND ----------

df_clean = df_raw.withColumn("label", when(col("income") == ">50K", 1).otherwise(0)).drop("income")

# COMMAND ----------

display(df_clean.select("education", "education_num").distinct().orderBy("education_num"))

# Here we see that the education_num is just a numerical value of education, so we remove the categorical columns education
df_clean = df_clean.drop("education")

# Due to size limitations, we remove this....
df_clean = df_clean.drop("native_country")

# COMMAND ----------

# Continue cleaning
total_rows = df_clean.count()

for c in df_clean.columns:
    dtype = df_clean.schema[c].dataType
    if isinstance(dtype, StringType):
        missing_count = df_clean.filter(
            (col(c).isNull()) |
            (col(c) == " ?") |
            (col(c) == "?") |
            (col(c) == "")
        ).count()
    else:
        missing_count = df_clean.filter(
            col(c).isNull()
        ).count()
    
    if missing_count > 0:
        print(c, missing_count)

# COMMAND ----------

def check_for_missing(df, print_count=True, print_res=False):
    total_missing = 0
    
    for c in df.columns:
        missing_df = df.filter(col(c).isNull())
        count = missing_df.count()
        
        if count > 0:
            total_missing += count
            if print_count:
                print(f"Column '{c}': {count} missing")
            
            if print_res:
                examples = missing_df.select(c).limit(5).collect()
                print(f"  Examples: {[r[0] for r in examples]}")
                
    return total_missing

def update_missing(df):
    for c in df.columns:
        if isinstance(df.schema[c].dataType, StringType):
                    df = df.withColumn(c, 
                        when(trim(col(c)).isin(["?", ""]), None)
                        .otherwise(col(c))
                    )
    return df

def drop_missing(df):
    df = df.na.drop()

    if check_for_missing(df, print_count=False) != 0: 
        print("Missing values still exist: Dropping")
        check_for_missing(df, print_res=True)
        raise ValueError("Missing values still exist")

    return df

def impute_mean(df):
    num_cols = [f.name for f in df.schema.fields if not isinstance(f.dataType, StringType)]
    if num_cols:
        imputer = Imputer(inputCols=num_cols, outputCols=num_cols).setStrategy("mean")
        try:
            model = imputer.fit(df)
            df = model.transform(df)
        except:
            pass # No nulls in num columns

    cat_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    mode_dict = {}
    
    for c in cat_cols:
        mode_row = df.filter(col(c).isNotNull()) \
                     .groupBy(c).count() \
                     .orderBy(col("count").desc()) \
                     .first()
        if mode_row:
            mode_dict[c] = mode_row[0]
    
    df = df.na.fill(mode_dict)

    if check_for_missing(df, print_count=False) != 0: 
        print("Missing values still exist: Mean imputation")
        check_for_missing(df, print_res=True)
        raise ValueError("Missing values still exist")

    return df

def impute_grouped_mode(df, target_col, group_cols):
    """
    Fills missing values in a CATEGORICAL 'target_col' using the MODE of 'group_cols'.
    Example: Fill missing 'Occupation' based on most common job for ('Education', 'Marital_Status').
    """
    print(f"Imputing Categorical {target_col} based on groups: {group_cols}")

    counts = df.filter(col(target_col).isNotNull()) \
               .groupBy(group_cols + [target_col]) \
               .agg(count("*").alias("cnt"))
    
    window_spec = Window.partitionBy(group_cols).orderBy(col("cnt").desc())
    
    # Select only the top row (Rank 1)
    group_modes = counts.withColumn("rank", row_number().over(window_spec)) \
                        .filter(col("rank") == 1) \
                        .select(group_cols + [col(target_col).alias("group_mode")])
    
    # 3. Join back and Fill
    df_joined = df.join(group_modes, on=group_cols, how="left")
    
    # 4. Fallback: Global Mode
    # If a group is totally empty, we need a global safety net
    global_mode = df.groupBy(target_col).count().orderBy(col("count").desc()).first()[0]
    
    df_filled = df_joined.withColumn(target_col, 
        coalesce(col(target_col), col("group_mode"), lit(global_mode))
    ).drop("group_mode")
    
    return df_filled

def predictive_grouping(df):
    # We know that the missing classes are workclass, occupation and native country from previous analysis
    # We can imagine at least education and sex being useful predictiors for workclass and occupatio
    job_group_keys = ["education_num", "sex"]
    for c_col in ["workclass", "occupation"]:
        if df.filter(col(c_col).isNull()).count() > 0:
            df = impute_grouped_mode(df, target_col=c_col, group_cols=job_group_keys)
        
    cat_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    df = df.na.fill({c: "Unknown" for c in cat_cols})

    if check_for_missing(df, print_count=False) != 0: 
        print("Missing values still exist: Predictive Grouping")
        check_for_missing(df, print_res=True)
        raise ValueError("Missing values still exist")

    return df

# COMMAND ----------

def build_pipeline(df):
    # There are some categorical columns that needs to be handled:
    cat_columns = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex']
    num_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

    # None of these have relations that makes one better than the other, so we do not need ordinal encoding here
    stages = []

    for c in cat_columns:
        indexer = StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="skip")
        encoder = OneHotEncoder(inputCol=c+"_index", outputCol=c+"_vec")

        stages += [indexer, encoder]

    num_assembler = VectorAssembler(inputCols=num_cols, outputCol="num_features_raw")
    stages.append(num_assembler)

    scaler = StandardScaler(inputCol="num_features_raw", outputCol="num_features_scaled", 
                                withStd=True, withMean=True)
    stages.append(scaler)

    final_inputs = [c + "_vec" for c in cat_columns] + ["num_features_scaled"]
    final_assembler = VectorAssembler(inputCols=final_inputs, outputCol="features")
    stages.append(final_assembler)

    return Pipeline(stages=stages)

def get_top_categories(df, col_name, top_k=5):
    """
    Scans the dataframe and returns a Python list of the Top K values.
    """
    # Group, Count, Sort Descending, Take Top K
    top_rows = df.groupBy(col_name).count().orderBy(col("count").desc()).limit(top_k).collect()
    
    # Extract just the strings
    top_values = [row[0] for row in top_rows]
    
    print(f"Learned Top {top_k} categories for '{col_name}': {top_values}")
    return top_values

def apply_cardinality_mask(df, col_name, allowed_values):
    """
    Forces the column to only contain values in 'allowed_values'.
    Everything else becomes 'Other'.
    """
    # Logic: If value is in the allowed list, keep it. Else -> "Other"
    return df.withColumn(col_name, 
        when(col(col_name).isin(allowed_values), col(col_name))
        .otherwise(lit("Other"))
    )

# COMMAND ----------

def evaluate_classifier(predictions, label_col="label", prediction_col="prediction"):
    """
    Prints a comprehensive classification report for PySpark models.
    """

    mc_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col)
    bc_evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction")
    
    # 2. Calculate Metrics
    accuracy = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "accuracy"})
    precision = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedPrecision"})
    recall = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedRecall"})
    f1 = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "f1"})
    auc = bc_evaluator.evaluate(predictions, {bc_evaluator.metricName: "areaUnderROC"})

    class_counts = predictions.groupBy(label_col).count().orderBy(label_col).collect()
    total_rows = predictions.count()
    
    print("--- Classification Report ---")
    print(f"Total Rows: {total_rows}")
    for row in class_counts:
        label = row[label_col]
        count = row['count']
        pct = (count / total_rows) * 100
        print(f"  - Class {label} count: {count} ({pct:.1f}%)")
    
    # 3. Print Report
    print("-" * 30)
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print(f"AUC (ROC): {auc:.4f}")
    print("-" * 30)

    cm = predictions.groupBy("label", "prediction").count().orderBy("label", "prediction")

    print("--- Confusion Matrix ---")
    print("Label = Truth, Prediction = Model Guess")
    cm.show()
    
    return {"accuracy": accuracy, "f1": f1, "auc": auc}

# COMMAND ----------

def test_impute_strat(train, val):

    prep_pipeline_standard = build_pipeline(df=train)
    prep_model_stanard = prep_pipeline_standard.fit(train)

    train = prep_model_stanard.transform(train)
    val = prep_model_stanard.transform(val)

    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    lr_model = lr.fit(train)
    predictions = lr_model.transform(val)

    metrics = evaluate_classifier(predictions)

    del lr_model, prep_model_stanard, prep_pipeline_standard, lr, predictions, train, val

# COMMAND ----------

train_raw, temp_df = df_clean.randomSplit([0.6, 0.4], seed=42)
val_raw, test_raw = temp_df.randomSplit([0.5, 0.5], seed=42)

#check_for_missing(test_raw)
#test_clean = drop_missing(test_raw)

# We investigate 3 strategies for handling missing data
train_raw = update_missing(train_raw)
val_raw = update_missing(val_raw)

# COMMAND ----------

# 1. Just removing the missing data
train_no_impute = drop_missing(train_raw)
val_no_impute = drop_missing(val_raw)

test_impute_strat(train=train_no_impute, val=val_no_impute)

del train_no_impute, val_no_impute

# COMMAND ----------



# COMMAND ----------

# 2. Impute with mean for numerical and the most common value for categorical
train_impute_mean = impute_mean(train_raw)
val_impute_mean = impute_mean(val_raw)

test_impute_strat(train=train_impute_mean, val=val_impute_mean)

del train_impute_mean, val_impute_mean

# COMMAND ----------

# 3. A "smarter" predictive grouping based on other columns
train_p_impute = predictive_grouping(train_raw)
val_p_impute = predictive_grouping(val_raw)

test_impute_strat(train=train_p_impute, val=val_p_impute)
