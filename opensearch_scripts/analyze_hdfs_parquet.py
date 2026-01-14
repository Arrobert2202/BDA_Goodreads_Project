from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Inspect_Goodreads_Data") \
    .getOrCreate()

def inspect_hdfs_parquet(path, label):
    print(f"\n--- Inspecting {label} ---")
    df = spark.read.parquet(path)
    df.printSchema()
    print(df.limit(5).toPandas().to_string(index=False))

inspect_hdfs_parquet("hdfs:///user/ubuntu/proiect_bda/output_nlp/parquet", "NLP Data")
inspect_hdfs_parquet("hdfs:///user/ubuntu/proiect_bda/output_recommender/parquet", "Recommender")
inspect_hdfs_parquet("hdfs:///user/ubuntu/proiect_bda/output_eda/parquet", "EDA")