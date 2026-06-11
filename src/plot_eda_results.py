import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, substring, col

spark = SparkSession.builder.appName("AvazuPlots").getOrCreate()

df = spark.read.csv("data/raw/train.csv", header=True, inferSchema=True)

df = df.withColumn(
    "hour_of_day",
    substring(col("hour").cast("string"), 7, 2)
)

# 1. Click distribution
click_dist = (
    df.groupBy("click")
    .count()
    .orderBy("click")
    .toPandas()
)

plt.figure(figsize=(6, 4))
plt.bar(click_dist["click"].astype(str), click_dist["count"])
plt.title("Click Distribution")
plt.xlabel("Click")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/click_distribution.png", dpi=300)
plt.close()

# 2. CTR by hour
ctr_hour = (
    df.groupBy("hour_of_day")
    .agg(
        count("*").alias("impressions"),
        avg("click").alias("ctr")
    )
    .orderBy("hour_of_day")
    .toPandas()
)

ctr_hour["ctr_percent"] = ctr_hour["ctr"] * 100

plt.figure(figsize=(9, 5))
plt.plot(ctr_hour["hour_of_day"], ctr_hour["ctr_percent"], marker="o")
plt.title("Click-Through Rate by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("CTR (%)")
plt.tight_layout()
plt.savefig("results/ctr_by_hour.png", dpi=300)
plt.close()

# 3. CTR by banner position
ctr_banner = (
    df.groupBy("banner_pos")
    .agg(
        count("*").alias("impressions"),
        avg("click").alias("ctr")
    )
    .orderBy("banner_pos")
    .toPandas()
)

ctr_banner["ctr_percent"] = ctr_banner["ctr"] * 100

plt.figure(figsize=(7, 5))
plt.bar(ctr_banner["banner_pos"].astype(str), ctr_banner["ctr_percent"])
plt.title("Click-Through Rate by Banner Position")
plt.xlabel("Banner Position")
plt.ylabel("CTR (%)")
plt.tight_layout()
plt.savefig("results/ctr_by_banner_position.png", dpi=300)
plt.close()

# 4. CTR by device type
ctr_device = (
    df.groupBy("device_type")
    .agg(
        count("*").alias("impressions"),
        avg("click").alias("ctr")
    )
    .orderBy("device_type")
    .toPandas()
)

ctr_device["ctr_percent"] = ctr_device["ctr"] * 100

plt.figure(figsize=(7, 5))
plt.bar(ctr_device["device_type"].astype(str), ctr_device["ctr_percent"])
plt.title("Click-Through Rate by Device Type")
plt.xlabel("Device Type")
plt.ylabel("CTR (%)")
plt.tight_layout()
plt.savefig("results/ctr_by_device_type.png", dpi=300)
plt.close()

spark.stop()

print("Saved plots:")
print("results/click_distribution.png")
print("results/ctr_by_hour.png")
print("results/ctr_by_banner_position.png")
print("results/ctr_by_device_type.png")