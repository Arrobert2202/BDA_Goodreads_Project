from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CheckData").getOrCreate()

path = "hdfs:///user/ubuntu/proiect_bda/output_nlp/parquet"

print(f"Readin data from: {path}")
df = spark.read.parquet(path)

print("--- SCHEMA ---")
df.printSchema()

print("--- DATA SAMPLE ---")
df.show(5)

print(f"--- TOTAL ROWS: {df.count()} ---")