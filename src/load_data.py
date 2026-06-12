from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AvazuCTR").getOrCreate()

df = spark.read.csv("data/raw/train.csv", header=True, inferSchema=True)

print("Data loaded")
print("Rows: {}, Columns: {}".format(df.count(), len(df.columns)))

df.printSchema()
df.show(5, truncate=False)

spark.stop()