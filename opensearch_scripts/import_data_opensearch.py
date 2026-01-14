import json
import requests
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Goodreads").getOrCreate()
df = spark.read.parquet("hdfs:///user/ubuntu/proiect_bda/output_eda/parquet")

def index_partition(partition):
  url = "http://172.16.30.224:9200/goodreads_catalog/_bulk"
  headers = {'Content-Type': 'application/x-ndjson'}

  bulk_data = ""
  for row in partition:
  index_line = {"index": {"_index": "goodreads_catalog", "_id": row["book_id"]}}
  data_line = {
      "title": row["title"],
      "description": row["description"],
      "average_rating": row["average_rating"],
      "similar_books": row["similar_books"]
  }
  bulk_data += json.dumps(index_line) + "\n" + json.dumps(data_line) + "\n"

  if bulk_data.count('\n') >= 1000:
      try:
          requests.post(url, data=bulk_data, headers=headers)
      except Exception as e:
          print(f"Error posting: {e}")
          bulk_data = ""

  if bulk_data:
    requests.post(url, data=bulk_data, headers=headers)

df.rdd.foreachPartition(index_partition)