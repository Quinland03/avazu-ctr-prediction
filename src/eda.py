from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, substring, col

spark = SparkSession.builder.appName("AvazuEDA").getOrCreate()

df = spark.read.csv("data/raw/train.csv", header=True, inferSchema=True)

print("Dataset shape")
print("Rows:", df.count())
print("Columns:", len(df.columns))

print("\nClick distribution")
df.groupBy("click").count().show()

print("\nOverall click-through rate")
df.select(avg("click").alias("ctr")).show()

df = df.withColumn("hour_of_day", substring(col("hour").cast("string"), 7, 2))

print("\nCTR by hour of day")
df.groupBy("hour_of_day").agg(
    count("*").alias("impressions"),
    avg("click").alias("ctr")
).orderBy("hour_of_day").show(24)

print("\nCTR by banner position")
df.groupBy("banner_pos").agg(
    count("*").alias("impressions"),
    avg("click").alias("ctr")
).orderBy("banner_pos").show()

print("\nCTR by device type")
df.groupBy("device_type").agg(
    count("*").alias("impressions"),
    avg("click").alias("ctr")
).orderBy("device_type").show()

spark.stop()