# Databricks notebook source
from pyspark.sql.functions import col, trim, when, lit, coalesce, row_number, count
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml import Pipeline

# COMMAND ----------

dataset_path = "/databricks-datasets/adult/adult.data"

# COMMAND ----------

df_raw = spark.read.format("csv").option("inferSchema", "true").load(dataset_path)

# COMMAND ----------

display(df_raw)

# COMMAND ----------

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

    if df.filter(col("native_country").isNull()).count() > 0:
            df = impute_grouped_mode(df, target_col="native_country", group_cols=["race"])
        
    cat_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    df = df.na.fill({c: "Unknown" for c in cat_cols})

    if check_for_missing(df, print_count=False) != 0: 
        print("Missing values still exist: Predictive Grouping")
        check_for_missing(df, print_res=True)
        raise ValueError("Missing values still exist")

    return df

# COMMAND ----------

train_raw, test_df = df_clean.randomSplit([0.6, 0.4], seed=42)

check_for_missing(test_raw)
test_clean = drop_missing(test_raw)

# We investigate 3 strategies for handling missing data
train_raw = update_missing(train_raw)
val_raw = update_missing(val_raw)

# 1. Just removing the missing data
train_no_impute = drop_missing(train_raw)
val_no_impute = drop_missing(val_raw)

# COMMAND ----------

# 2. Impute with mean for numerical and the most common value for categorical
train_impute_mean = impute_mean(train_raw)
val_impute_mean = impute_mean(val_raw)

# 3. A "smarter" predictive grouping based on other columns
train_p_impute = predictive_grouping(train_raw)
val_p_impute = predictive_grouping(val_raw)


# COMMAND ----------

def build_pipeline():
    # There are some categorical columns that needs to be handled:
    cat_columns = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

    # None of these have relations that makes one better than the other, so we do not need ordinal encoding here
    stages = []

    for c in cat_columns:
        indexer = StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep")
        encoder = OneHotEncoder(inputCol=c+"_index", outputCol=c+"_vec")

        stages += [indexer, encoder]

    num_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

    assembler_inputs = [c + "_vec" for c in cat_columns] + num_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    stages.append(assembler)

    return stages

# COMMAND ----------



