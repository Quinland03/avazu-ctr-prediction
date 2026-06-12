from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("AvazuLogisticRegression").getOrCreate()

df = spark.read.csv("data/raw/train.csv", header=True, inferSchema=True)

df = df.withColumn("hour_of_day", substring(col("hour").cast("string"), 7, 2))

# Use 5% sample first so your laptop can handle it
df = df.sample(False, 0.05, seed=42)

features = [
    "banner_pos",
    "site_category",
    "app_category",
    "device_type",
    "device_conn_type",
    "C1",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "hour_of_day"
]

df = df.select(["click"] + features).dropna()

categorical_cols = ["site_category", "app_category", "hour_of_day"]

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(
        inputCol=f"{c}_idx",
        outputCol=f"{c}_vec"
    )
    for c in categorical_cols
]

numeric_cols = [
    "banner_pos",
    "device_type",
    "device_conn_type",
    "C1",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21"
]

assembler = VectorAssembler(
    inputCols=numeric_cols + [f"{c}_vec" for c in categorical_cols],
    outputCol="features"
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="click",
    maxIter=20
)

pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

train, test = df.randomSplit([0.8, 0.2], seed=42)

print("Training rows:", train.count())
print("Testing rows:", test.count())

model = pipeline.fit(train)

predictions = model.transform(test)

from pyspark.sql.functions import col, log, when
from pyspark.ml.functions import vector_to_array

eps = 1e-15

preds = predictions.select(
    col("click").alias("label"),
    vector_to_array(col("probability"))[1].alias("p")
)

preds = preds.withColumn(
    "p",
    when(col("p") < eps, eps)
    .when(col("p") > 1 - eps, 1 - eps)
    .otherwise(col("p"))
)

logloss_df = preds.withColumn(
    "logloss",
    -(col("label") * log(col("p")) +
      (1 - col("label")) * log(1 - col("p")))
)

logloss = logloss_df.agg(
    {"logloss": "avg"}
).collect()[0][0]

evaluator = BinaryClassificationEvaluator(
    labelCol="click",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)

print("\n========================")
print("Logistic Regression Results")
print("========================")
print(f"AUC: {auc:.6f}")
print(f"Log Loss: {logloss:.6f}")
print("========================\n")

predictions.select(
    "click",
    "probability",
    "prediction"
).show(10, truncate=False)

spark.stop()