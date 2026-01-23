from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("AirflowSparkYarn")
    .getOrCreate()
)

# Create some data
df = spark.range(0, 100)

# Write to HDFS
df.write.mode("overwrite").parquet("hdfs:///user/airflow/example_output")

print("Wrote data to HDFS at /user/airflow/example_output")

spark.stop()
