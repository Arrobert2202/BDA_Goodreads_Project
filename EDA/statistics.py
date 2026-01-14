from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, count, when, size, round as spark_round, lit, percentile_approx, sum as spark_sum
from pyspark.sql.functions import mean, stddev, min as spark_min, max as spark_max, explode
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType


SAMPLE_BASE = "hdfs:///user/ubuntu/proiect_bda/output_eda/sample_json"
FULL_BASE = "hdfs:///user/ubuntu/proiect_bda/output_eda/parquet"

OUTPUT_DIR = FULL_BASE + "/output_parquet_stats"

spark = SparkSession.builder \
    .appName("Goodreads_Master_ETL") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

df_books = spark.read.parquet(f"{FULL_BASE}")


######################################
# General counts
######################################

# Calculate total number of books and distinct books by ID
total_books = df_books.count()
distinct_books = df_books.select("book_id").distinct().count()
print(f"\n=== BOOKS DATA SUMMARY ===")
print(f"Total Books: {total_books}")
print(f"Distinct Books by book_id: {distinct_books}")

# Analyze missing or empty data for critical fields
missing_stats = df_books.select(
    count(when(col("description").isNull() | (col("description") == ""), True)).alias("missing_description"),
    count(when(col("authors").isNull() | (size(col("authors")) == 0), True)).alias("missing_authors"),
    count(when(col("average_rating").isNull(), True)).alias("missing_average_rating"),
    count(when(col("popular_shelves").isNull() | (size(col("popular_shelves")) == 0), True)).alias("missing_popular_shelves")
).collect()[0]

print(f"\n=== MISSING DATA ANALYSIS ===")
print(f"Missing description: {missing_stats['missing_description']} ({missing_stats['missing_description']/total_books*100:.2f}%)")
print(f"Missing authors: {missing_stats['missing_authors']} ({missing_stats['missing_authors']/total_books*100:.2f}%)")
print(f"Missing average_rating: {missing_stats['missing_average_rating']} ({missing_stats['missing_average_rating']/total_books*100:.2f}%)")
print(f"Missing popular_shelves: {missing_stats['missing_popular_shelves']} ({missing_stats['missing_popular_shelves']/total_books*100:.2f}%)")


######################################
# Page counts
######################################

# Filter books with valid page counts
df_pages = df_books.filter((col("num_pages").isNotNull()) & (col("num_pages") > 0))

# Calculate page statistics
page_stats = df_pages.select(
    spark_round(mean("num_pages"), 2).alias("mean_pages"),
    spark_round(stddev("num_pages"), 2).alias("std_pages"),
    spark_min("num_pages").alias("min_pages"),
    spark_max("num_pages").alias("max_pages"),
    percentile_approx("num_pages", 0.25).alias("q1_pages"),
    percentile_approx("num_pages", 0.5).alias("median_pages"),
    percentile_approx("num_pages", 0.75).alias("q3_pages")
).collect()[0]

print("\n=== PAGE STATISTICS ===")
print(f"Mean pages: {page_stats['mean_pages']}")
print(f"Median pages: {page_stats['median_pages']}")
print(f"Std dev: {page_stats['std_pages']}")
print(f"Min pages: {page_stats['min_pages']}")
print(f"Max pages: {page_stats['max_pages']}")
print(f"Q1 (25th percentile): {page_stats['q1_pages']}")
print(f"Q3 (75th percentile): {page_stats['q3_pages']}")

# Categorize books by page count into meaningful ranges
page_distribution = df_pages \
    .withColumn("page_category", 
        when(col("num_pages") < 100, "Very Short (<100)")
        .when(col("num_pages") < 200, "Short (100-200)")
        .when(col("num_pages") < 300, "Medium-Short (200-300)")
        .when(col("num_pages") < 400, "Medium (300-400)")
        .when(col("num_pages") < 500, "Medium-Long (400-500)")
        .when(col("num_pages") < 700, "Long (500-700)")
        .otherwise("Very Long (700+)")
    ) \
    .groupBy("page_category") \
    .agg(count("*").alias("book_count")) \
    .withColumn("percentage", 
        spark_round((col("book_count") / lit(df_pages.count())) * 100, 2)
    ) \
    .orderBy(
        when(col("page_category") == "Very Short (<100)", 1)
        .when(col("page_category") == "Short (100-200)", 2)
        .when(col("page_category") == "Medium-Short (200-300)", 3)
        .when(col("page_category") == "Medium (300-400)", 4)
        .when(col("page_category") == "Medium-Long (400-500)", 5)
        .when(col("page_category") == "Long (500-700)", 6)
        .otherwise(7)
    )

print("\n=== PAGE DISTRIBUTION ===")
page_distribution.show(truncate=False)
page_distribution.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/page_distribution")


######################################
# Series analysis
######################################

# Determine how many books are part of a series vs standalone
books_with_series = df_books.filter((col("series").isNotNull()) & (size(col("series")) > 0)).count()
books_without_series = total_books - books_with_series

print("\n=== SERIES OVERVIEW ===")
print(f"Books in Series: {books_with_series} ({books_with_series/total_books*100:.2f}%)")
print(f"Standalone Books: {books_without_series} ({books_without_series/total_books*100:.2f}%)")

series_count_distribution = df_books \
    .filter(col("series").isNotNull()) \
    .withColumn("series_count", size(col("series"))) \
    .withColumn("series_category",
        when(col("series_count") == 0, "No Series")
        .when(col("series_count") == 1, "Single Series")
        .when(col("series_count") == 2, "2 Series")
        .when(col("series_count") == 3, "3 Series")
        .otherwise("4+ Series")
    ) \
    .groupBy("series_category") \
    .agg(count("*").alias("book_count")) \
    .withColumn("percentage",
        spark_round((col("book_count") / lit(total_books)) * 100, 2)
    ) \
    .orderBy(
        when(col("series_category") == "No Series", 1)
        .when(col("series_category") == "Single Series", 2)
        .when(col("series_category") == "2 Series", 3)
        .when(col("series_category") == "3 Series", 4)
        .otherwise(5)
    )

print("\n=== SERIES COUNT DISTRIBUTION ===")
series_count_distribution.show(truncate=False)
series_count_distribution.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/series_count_distribution")


######################################
# Avg Rating distribution
######################################

# Calculate books with valid ratings
total_with_ratings = total_books - missing_stats['missing_average_rating']

# Distribution of book quality as perceived by readers
rating_distribution = df_books \
    .filter(col("average_rating").isNotNull()) \
    .withColumn("rating_bucket", 
        when(col("average_rating") < 1.0, "0.0-1.0")
        .when(col("average_rating") < 2.0, "1.0-2.0")
        .when(col("average_rating") < 3.0, "2.0-3.0")
        .when(col("average_rating") < 4.0, "3.0-4.0")
        .when(col("average_rating") < 5.0, "4.0-5.0")
        .otherwise("5.0")
    ) \
    .groupBy("rating_bucket") \
    .agg(count("*").alias("book_count")) \
    .withColumn("percentage", 
        spark_round((col("book_count") / lit(total_with_ratings)) * 100, 2)
    ) \
    .orderBy("rating_bucket")

print("\n=== AVERAGE RATING DISTRIBUTION ===")
rating_distribution.show()
rating_distribution.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/rating_distribution")

rating_stats = df_books \
    .filter(col("average_rating").isNotNull()) \
    .select(
        spark_round(mean("average_rating"), 2).alias("mean_rating"),
        spark_round(stddev("average_rating"), 2).alias("std_rating"),
        spark_min("average_rating").alias("min_rating"),
        spark_max("average_rating").alias("max_rating")
    )

print("\n=== RATING STATISTICS ===")
rating_stats.show()
rating_stats.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/rating_stats")


######################################
# Book engagement 
######################################

df_engagement = df_books.filter(col("ratings_count").isNotNull())

# ratings_count as a proxy for popularity
engagement_stats = df_engagement.select(
    spark_round(mean("ratings_count"), 2).alias("mean_ratings"),
    spark_round(stddev("ratings_count"), 2).alias("std_ratings"),
    spark_min("ratings_count").alias("min_ratings"),
    spark_max("ratings_count").alias("max_ratings"),
    percentile_approx("ratings_count", 0.25).alias("q1_ratings"),
    percentile_approx("ratings_count", 0.5).alias("median_ratings"),
    percentile_approx("ratings_count", 0.75).alias("q3_ratings")
).collect()[0]

print("\n=== ENGAGEMENT STATISTICS ===")
print(f"Mean ratings: {engagement_stats['mean_ratings']}")
print(f"Median ratings: {engagement_stats['median_ratings']}")
print(f"Std dev: {engagement_stats['std_ratings']}")
print(f"Min ratings: {engagement_stats['min_ratings']}")
print(f"Max ratings: {engagement_stats['max_ratings']}")
print(f"Q1 (25th percentile): {engagement_stats['q1_ratings']}")
print(f"Q3 (75th percentile): {engagement_stats['q3_ratings']}")

engagement_distribution = df_engagement \
    .withColumn("engagement_level", 
        when(col("ratings_count") < 100, "Very Low (<100)")
        .when(col("ratings_count") < 1000, "Low (100-1K)")
        .when(col("ratings_count") < 10000, "Medium (1K-10K)")
        .when(col("ratings_count") < 100000, "High (10K-100K)")
        .when(col("ratings_count") < 1000000, "Very High (100K-1M)")
        .otherwise("Extremely High (1M+)")
    ) \
    .groupBy("engagement_level") \
    .agg(count("*").alias("book_count")) \
    .withColumn("percentage", 
        spark_round((col("book_count") / lit(df_engagement.count())) * 100, 2)
    ) \
    .orderBy(
        when(col("engagement_level") == "Very Low (<100)", 1)
        .when(col("engagement_level") == "Low (100-1K)", 2)
        .when(col("engagement_level") == "Medium (1K-10K)", 3)
        .when(col("engagement_level") == "High (10K-100K)", 4)
        .when(col("engagement_level") == "Very High (100K-1M)", 5)
        .otherwise(6)
    )

print("\n=== ENGAGEMENT DISTRIBUTION ===")
engagement_distribution.show(truncate=False)
engagement_distribution.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/engagement_distribution")

######################################
# Book engagement VS Rating quality
######################################

# identify if highly-rated books also have more engagement
quality_vs_engagement = df_books \
    .filter((col("average_rating").isNotNull()) & (col("ratings_count").isNotNull())) \
    .withColumn("rating_quality", 
        when(col("average_rating") < 3.0, "Low (< 3.0)")
        .when(col("average_rating") < 4.0, "Medium (3.0-4.0)")
        .otherwise("High (4.0+)")
    ) \
    .withColumn("engagement_level", 
        when(col("ratings_count") < 100, "Low (<100)")
        .when(col("ratings_count") < 1000, "Medium (100-1K)")
        .when(col("ratings_count") < 10000, "High (1K-10K)")
        .otherwise("Very High (10K+)")
    ) \
    .groupBy("rating_quality", "engagement_level") \
    .agg(count("*").alias("book_count")) \
    .orderBy("rating_quality", "engagement_level")

print("\n=== RATING QUALITY VS ENGAGEMENT ===")
quality_vs_engagement.show(20, truncate=False)
quality_vs_engagement.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/quality_vs_engagement")


######################################
# Shelves Analysis
######################################

books_with_shelves = df_books.filter((col("popular_shelves").isNotNull()) & (size(col("popular_shelves")) > 0)).count()
books_without_shelves = total_books - books_with_shelves

print("\n=== SHELVES OVERVIEW ===")
print(f"Books with Shelves: {books_with_shelves} ({books_with_shelves/total_books*100:.2f}%)")
print(f"Books without Shelves: {books_without_shelves} ({books_without_shelves/total_books*100:.2f}%)")

shelves_count_distribution = df_books \
    .filter(col("popular_shelves").isNotNull()) \
    .withColumn("shelves_count", size(col("popular_shelves"))) \
    .withColumn("shelves_category",
        when(col("shelves_count") == 0, "No Shelves")
        .when(col("shelves_count").between(1, 5), "1-5 Shelves")
        .when(col("shelves_count").between(6, 10), "6-10 Shelves")
        .when(col("shelves_count").between(11, 20), "11-20 Shelves")
        .when(col("shelves_count").between(21, 50), "21-50 Shelves")
        .otherwise("50+ Shelves")
    ) \
    .groupBy("shelves_category") \
    .agg(count("*").alias("book_count")) \
    .withColumn("percentage",
        spark_round((col("book_count") / lit(total_books)) * 100, 2)
    ) \
    .orderBy(
        when(col("shelves_category") == "No Shelves", 1)
        .when(col("shelves_category") == "1-5 Shelves", 2)
        .when(col("shelves_category") == "6-10 Shelves", 3)
        .when(col("shelves_category") == "11-20 Shelves", 4)
        .when(col("shelves_category") == "21-50 Shelves", 5)
        .otherwise(6)
    )

print("\n=== SHELVES COUNT DISTRIBUTION ===")
shelves_count_distribution.show(truncate=False)
shelves_count_distribution.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/shelves_count_distribution")

avg_shelves = df_books \
    .filter((col("popular_shelves").isNotNull()) & (size(col("popular_shelves")) > 0)) \
    .select(spark_round(mean(size(col("popular_shelves"))), 2).alias("avg_shelves_per_book")) \
    .collect()[0]

print(f"\nAverage shelves per book: {avg_shelves['avg_shelves_per_book']}")


######################################
# Top 10 Shelves
######################################

shelf_popularity = df_books \
    .filter((col("popular_shelves").isNotNull()) & (size(col("popular_shelves")) > 0)) \
    .select(
        col("book_id"),
        explode(col("popular_shelves")).alias("shelf")
    ) \
    .select(
        col("book_id"),
        col("shelf.name").alias("shelf_name"),
        col("shelf.count").cast("int").alias("shelf_count")
    ) \
    .groupBy("shelf_name") \
    .agg(
        count("book_id").alias("books_with_shelf"),
        spark_sum("shelf_count").alias("total_shelf_count")
    ) \
    .orderBy(col("books_with_shelf").desc())

print("\n=== TOP 10 MOST POPULAR SHELF NAMES (by book count) ===")
shelf_popularity.show(10, truncate=False)
shelf_popularity.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/shelf_popularity")

most_used_shelves = shelf_popularity.orderBy(col("total_shelf_count").desc()).limit(10)

print("\n=== TOP 10 MOST USED SHELVES (by total usage count) ===")
most_used_shelves.show(10, truncate=False)
most_used_shelves.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/most_used_shelves")

######################################
# Similar Books Analysis
######################################

books_with_similar = df_books.filter((col("similar_books").isNotNull()) & (size(col("similar_books")) > 0)).count()
books_without_similar = total_books - books_with_similar

print("\n=== SIMILAR BOOKS OVERVIEW ===")
print(f"Total Books: {total_books}")
print(f"Books with Similar Books: {books_with_similar} ({books_with_similar/total_books*100:.2f}%)")
print(f"Books without Similar Books: {books_without_similar} ({books_without_similar/total_books*100:.2f}%)")

# Average similar books per book
avg_similar = df_books \
    .filter((col("similar_books").isNotNull()) & (size(col("similar_books")) > 0)) \
    .select(spark_round(mean(size(col("similar_books"))), 2).alias("avg_similar_books")) \
    .collect()[0]

print(f"Average similar books per book: {avg_similar['avg_similar_books']}")

# Similar books count distribution
similar_books_distribution = df_books \
    .filter(col("similar_books").isNotNull()) \
    .withColumn("similar_count", size(col("similar_books"))) \
    .withColumn("similar_category",
        when(col("similar_count") == 0, "No Similar Books")
        .when(col("similar_count").between(1, 5), "1-5 Similar")
        .when(col("similar_count").between(6, 10), "6-10 Similar")
        .when(col("similar_count").between(11, 20), "11-20 Similar")
        .when(col("similar_count").between(21, 50), "21-50 Similar")
        .otherwise("50+ Similar")
    ) \
    .groupBy("similar_category") \
    .agg(count("*").alias("book_count")) \
    .withColumn("percentage",
        spark_round((col("book_count") / lit(total_books)) * 100, 2)
    ) \
    .orderBy(
        when(col("similar_category") == "No Similar Books", 1)
        .when(col("similar_category") == "1-5 Similar", 2)
        .when(col("similar_category") == "6-10 Similar", 3)
        .when(col("similar_category") == "11-20 Similar", 4)
        .when(col("similar_category") == "21-50 Similar", 5)
        .otherwise(6)
    )

print("\n=== SIMILAR BOOKS COUNT DISTRIBUTION ===")
similar_books_distribution.show(truncate=False)
similar_books_distribution.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/similar_books_distribution")

# Correlation between number of similar books and book performance
similar_vs_performance = df_books \
    .filter(
        (col("similar_books").isNotNull()) &
        (col("average_rating").isNotNull()) &
        (col("ratings_count").isNotNull())
    ) \
    .withColumn("similar_count", size(col("similar_books"))) \
    .withColumn("similar_category",
        when(col("similar_count") == 0, "No Similar Books")
        .when(col("similar_count").between(1, 10), "1-10 Similar")
        .when(col("similar_count").between(11, 20), "11-20 Similar")
        .when(col("similar_count").between(21, 50), "21-50 Similar")
        .otherwise("50+ Similar")
    ) \
    .groupBy("similar_category") \
    .agg(
        count("*").alias("book_count"),
        spark_round(mean("average_rating"), 2).alias("avg_rating"),
        spark_round(mean("ratings_count"), 2).alias("avg_engagement"),
        spark_round(mean("similar_count"), 2).alias("avg_similar_count")
    ) \
    .orderBy(
        when(col("similar_category") == "No Similar Books", 1)
        .when(col("similar_category") == "1-10 Similar", 2)
        .when(col("similar_category") == "11-20 Similar", 3)
        .when(col("similar_category") == "21-50 Similar", 4)
        .otherwise(5)
    )

print("\n=== SIMILAR BOOKS COUNT VS BOOK PERFORMANCE ===")
similar_vs_performance.show(truncate=False)
similar_vs_performance.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/similar_vs_performance")

# books that appear most frequently in other books' similar_books lists
most_referenced = df_books \
    .filter((col("similar_books").isNotNull()) & (size(col("similar_books")) > 0)) \
    .select(
        col("book_id").alias("source_book_id"),
        explode(col("similar_books")).alias("similar_book_id")
    ) \
    .groupBy("similar_book_id") \
    .agg(count("source_book_id").alias("referenced_count")) \
    .orderBy(col("referenced_count").desc()) \
    .limit(50)

print("\n=== TOP 20 MOST REFERENCED BOOKS (appearing as similar books) ===")
most_referenced.show(20, truncate=False)
most_referenced.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/most_referenced_books")


###########################################################
# KeyWords Frequency Analysis in Descriptions
###########################################################

keywords = [
    "love","war","mystery","fantasy","magic","science","history",
    "biography","business","psychology","self-help",
    "adventure","journey","family","friendship","survival",
    "dark","emotional","inspiring","funny",
    "hero","villain","detective",
    "young","adult","children",
    "future","past","world","society",
    "bestselling","award"]

print("\n=== KEYWORD ANALYSIS IN DESCRIPTIONS ===")

# Filter to only books that have non-empty descriptions
df_with_desc = df_books.filter(
    (col("description").isNotNull()) & 
    (col("description") != "")
)
total_with_desc = df_with_desc.count()
print(f"Books with descriptions: {total_with_desc}")

# Iterate through each keyword to count occurrences in descriptions
keyword_results = []
for keyword in keywords:
    # Count books where the keyword appears in the description (case-insensitive)
    count_with_keyword = df_with_desc.filter(
        lower(col("description")).contains(keyword.lower())
    ).count()
    
    percentage = (count_with_keyword / total_with_desc * 100) if total_with_desc > 0 else 0
    keyword_results.append((keyword, count_with_keyword, round(percentage, 2)))

# Sort by frequency (descending)
keyword_results.sort(key=lambda x: x[1], reverse=True)

# Create DataFrame from keyword results

keyword_schema = StructType([
    StructField("keyword", StringType(), True),
    StructField("book_count", IntegerType(), True),
    StructField("percentage", DoubleType(), True)
])

keyword_df = spark.createDataFrame(keyword_results, keyword_schema)

print("\n=== TOP KEYWORDS IN DESCRIPTIONS (sorted by frequency) ===")
keyword_df.show(len(keywords), truncate=False)

print("\n=== TOP 10 MOST COMMON KEYWORDS ===")
keyword_df.limit(10).show(truncate=False)

print("\n=== 10 LEAST COMMON KEYWORDS ===")
keyword_df.orderBy(col("book_count").asc()).limit(10).show(truncate=False)

# Create binary indicator columns for each keyword (1 if present, 0 if not)
multi_keyword_presence = df_with_desc.select("book_id", "description")
for keyword in keywords:
    multi_keyword_presence = multi_keyword_presence.withColumn(
        f"has_{keyword}",
        when(lower(col("description")).contains(keyword.lower()), 1).otherwise(0)
    )

# Sum all keyword indicators to get total keyword count per book
keyword_columns = [f"has_{kw}" for kw in keywords]
multi_keyword_presence = multi_keyword_presence.withColumn(
    "total_keywords",
    sum([col(kw_col) for kw_col in keyword_columns])
)

keyword_count_distribution = multi_keyword_presence \
    .withColumn("keyword_category",
        when(col("total_keywords") == 0, "0 keywords")
        .when(col("total_keywords").between(1, 2), "1-2 keywords")
        .when(col("total_keywords").between(3, 5), "3-5 keywords")
        .when(col("total_keywords").between(6, 10), "6-10 keywords")
        .when(col("total_keywords").between(11, 15), "11-15 keywords")
        .otherwise("15+ keywords")
    ) \
    .groupBy("keyword_category") \
    .agg(count("*").alias("book_count")) \
    .withColumn("percentage",
        spark_round((col("book_count") / lit(total_with_desc)) * 100, 2)
    ) \
    .orderBy(
        when(col("keyword_category") == "0 keywords", 1)
        .when(col("keyword_category") == "1-2 keywords", 2)
        .when(col("keyword_category") == "3-5 keywords", 3)
        .when(col("keyword_category") == "6-10 keywords", 4)
        .when(col("keyword_category") == "11-15 keywords", 5)
        .otherwise(6)
    )

print("\n=== KEYWORD DENSITY IN DESCRIPTIONS ===")
keyword_count_distribution.show(truncate=False)
keyword_count_distribution.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/keyword_count_distribution")

avg_keywords = multi_keyword_presence.select(
    spark_round(mean("total_keywords"), 2).alias("avg_keywords_per_book")
).collect()[0]
print(f"\nAverage keywords per book description: {avg_keywords['avg_keywords_per_book']}")

spark.stop()