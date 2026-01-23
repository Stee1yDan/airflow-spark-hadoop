from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AirflowSparkYarn").getOrCreate()

df = spark.range(0, 100)
print(df.count())

spark.stop()
