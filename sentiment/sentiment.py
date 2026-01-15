import sys
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, avg, count
from pyspark.sql.types import FloatType, StringType
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

HDFS_BASE = "hdfs:///user/ubuntu/proiect_bda"
INPUT_PATH = f"{HDFS_BASE}/output_nlp/parquet"
OUTPUT_DETAILED = f"{HDFS_BASE}/output_nlp_final"
OUTPUT_INSIGHTS = f"{HDFS_BASE}/output_nlp_insights" 

def clean_text(text):
        if not text: return ""
        return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

def get_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

spark = SparkSession.builder \
    .appName("Goodreads_Sentiment_Algorithm") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n>>> Sentiment Analysis Job Started\n")

try:
    print(f">>> 1. Reading data from: {INPUT_PATH}")
    df = spark.read.parquet(INPUT_PATH)

    clean_udf = udf(clean_text, StringType())
    sentiment_udf = udf(get_sentiment, FloatType())

    print(">>> 2. Calculating Sentiment Scores")
    
    df_processed = df.withColumn("clean_text", clean_udf(col("review_text"))) \
                     .withColumn("sentiment_score", sentiment_udf(col("clean_text")))

    df_final = df_processed.withColumn("sentiment_label", 
        when(col("sentiment_score") >= 0.1, "POSITIVE")
        .when(col("sentiment_score") <= -0.1, "NEGATIVE")
        .otherwise("NEUTRAL")
    )

    df_final.cache()

    print(">>> 3. Aggregating Data (Rating vs Sentiment)")

    df_insights = df_final.groupBy("book_id", "title") \
        .agg(
            avg("sentiment_score").alias("avg_sentiment"),
            avg("rating").alias("avg_rating"),
            count("sentiment_score").alias("review_count")
        ) \
        .filter(col("review_count") > 20) \
        .orderBy(col("avg_sentiment").desc())

    print(f">>> 4. Saving Results to HDFS")
    
    df_final.drop("clean_text").write.mode("overwrite").parquet(f"{OUTPUT_DETAILED}/parquet")
    df_insights.write.mode("overwrite").parquet(f"{OUTPUT_INSIGHTS}/parquet")

    print(">>> 5. Exporting JSON for Visualization(Opensearch)")

    df_insights.limit(1000).coalesce(1) \
        .write.mode("overwrite").json(f"{OUTPUT_INSIGHTS}/json_top_books")

    df_final.select("title", "sentiment_label", "sentiment_score") \
        .limit(20000).coalesce(1) \
        .write.mode("overwrite").json(f"{OUTPUT_DETAILED}/json_reviews")

    print(">>> 6. Generating visualizations")

    pdf_pie = df_final.groupBy("sentiment_label").count().toPandas()
    
    plt.figure(figsize=(6, 6))
    plt.pie(pdf_pie['count'], labels=pdf_pie['sentiment_label'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Sentiment Distribution')
    plt.savefig('pie_chart.png')
    print("Generated: pie_chart.png")

    pdf_bar = df_insights.limit(10).toPandas()
    
    plt.figure(figsize=(10, 6))
    plt.barh(pdf_bar['title'], pdf_bar['avg_sentiment'], color='green')
    plt.xlabel('Sentiment Score')
    plt.title('Top 10 Books by Sentiment')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('bar_chart.png')
    print("Generated: bar_chart.png")

    print("\n>>> Job Finished Successfully")

except Exception as e:
    print(f"\n!!! Error: {e}")
    sys.exit(1)

spark.stop()