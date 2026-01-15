import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, size

HDFS_BASE = "hdfs:///user/ubuntu/proiect_bda"
INPUT_DIR = f"{HDFS_BASE}/input"
OUTPUT_DIR_NLP = f"{HDFS_BASE}/output_nlp"
OUTPUT_DIR_REC = f"{HDFS_BASE}/output_recommender"
OUTPUT_DIR_EDA = f"{HDFS_BASE}/output_eda"

spark = SparkSession.builder \
    .appName("Goodreads_Master_ETL") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print(f">>> Searching for files in: {INPUT_DIR}")

try:
    print(">>> Reading and processing books metadata")
    df_books = spark.read.json(f"{INPUT_DIR}/*books*.json") \
        .select(
            col("book_id"),
            col("title"),
            col("authors"),
            col("description"),
            col("average_rating").cast("float"),
            col("num_pages").cast("int"),
            col("ratings_count").cast("int"),
            col("text_reviews_count").cast("int"),
            col("popular_shelves"),
            col("series"),
            col("similar_books")
        )

    print(">>> Reading and processing reviews data")
    df_reviews = spark.read.json(f"{INPUT_DIR}/*reviews*.json") \
        .select(
            col("book_id"),
            col("user_id"),
            col("review_text"),
            col("rating").cast("int"),
            col("n_votes").cast("int"),
            col("date_added")
        )

except Exception as e:
    print(f"\n!!! Error reading input data from HDFS.")
    sys.exit(1)

print(">>> Joining Reviews with Books")
df_full = df_reviews.join(df_books, "book_id", "inner")

print(f">>> Total rows processed: {df_full.count()}")

print(">>> Generating sentiment analysis data")
df_nlp = df_full \
    .filter(col("review_text").isNotNull()) \
    .filter(length(col("review_text")) > 10) \
    .select("book_id", "user_id", "review_text", "rating", "n_votes", "title")

print(">>> Generating recommender system data")
df_rec = df_full \
    .filter(col("rating") > 0) \
    .select(
        "user_id", "book_id", "rating", "title", 
        "popular_shelves", "series", "similar_books", 
        "description", "average_rating"
    )

print(">>> Generating statistics data for EDA")
df_eda = df_books \
    .filter(col("ratings_count") > 10) \
    .filter(col("num_pages") > 0)

print(">>> Saving results to HDFS (Parquet format)")

df_nlp.write.mode("overwrite").parquet(f"{OUTPUT_DIR_NLP}/parquet")
df_rec.write.mode("overwrite").parquet(f"{OUTPUT_DIR_REC}/parquet")
df_eda.write.mode("overwrite").parquet(f"{OUTPUT_DIR_EDA}/parquet")

print(">>> Exporting JSON Samples")
df_nlp.limit(50000).coalesce(1).write.mode("overwrite").json(f"{OUTPUT_DIR_NLP}/sample_json")

df_eda.limit(50000).coalesce(1).write.mode("overwrite").json(f"{OUTPUT_DIR_EDA}/sample_json")

print("\n>>> Job Finished Successfully")
print("HDFS Output Locations:")
print(f"1. NLP: {OUTPUT_DIR_NLP}")
print(f"2. Recommender: {OUTPUT_DIR_REC}")
print(f"3. EDA: {OUTPUT_DIR_EDA}")

spark.stop()