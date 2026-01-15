from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, count as spark_count, when, size, round as spark_round, lit, percentile_approx, sum as spark_sum
from pyspark.sql.functions import mean, stddev, min as spark_min, max as spark_max
from pyspark.sql.functions import explode
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import os


SAMPLE_BASE = "hdfs:///user/ubuntu/proiect_bda/output_eda/sample_json"
FULL_BASE = "hdfs:///user/ubuntu/proiect_bda/output_eda/parquet"

OUTPUT_DIR = "./output_parquet_stats"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

total_books = df_books.count()
distinct_books = df_books.select("book_id").distinct().count()
print(f"\n=== BOOKS DATA SUMMARY ===")
print(f"Total Books: {total_books}")
print(f"Distinct Books by book_id: {distinct_books}")

missing_stats = df_books.select(
    spark_count(when(col("description").isNull() | (col("description") == ""), True)).alias("missing_description"),
    spark_count(when(col("authors").isNull() | (size(col("authors")) == 0), True)).alias("missing_authors"),
    spark_count(when(col("average_rating").isNull(), True)).alias("missing_average_rating"),
    spark_count(when(col("popular_shelves").isNull() | (size(col("popular_shelves")) == 0), True)).alias("missing_popular_shelves")
).collect()[0]

print(f"\n=== MISSING DATA ANALYSIS ===")
print(f"Missing description: {missing_stats['missing_description']} ({missing_stats['missing_description']/total_books*100:.2f}%)")
print(f"Missing authors: {missing_stats['missing_authors']} ({missing_stats['missing_authors']/total_books*100:.2f}%)")
print(f"Missing average_rating: {missing_stats['missing_average_rating']} ({missing_stats['missing_average_rating']/total_books*100:.2f}%)")
print(f"Missing popular_shelves: {missing_stats['missing_popular_shelves']} ({missing_stats['missing_popular_shelves']/total_books*100:.2f}%)")


######################################
# Page counts
######################################

df_pages = df_books.filter((col("num_pages").isNotNull()) & (col("num_pages") > 0))

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
    .agg(spark_count("*").alias("book_count")) \
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

page_dist_data = page_distribution.collect()
categories = [row['page_category'] for row in page_dist_data]
counts = [row['book_count'] for row in page_dist_data]
percentages = [row['percentage'] for row in page_dist_data]

plt.figure(figsize=(12, 6))
plt.bar(categories, counts, color='steelblue', edgecolor='black')
plt.xlabel('Page Category', fontsize=12)
plt.ylabel('Book Count', fontsize=12)
plt.title('Distribution of Books by Page Count', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/page_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Bar chart saved to {OUTPUT_DIR}/page_distribution.png")


######################################
# Series analysis
######################################


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
    .agg(spark_count("*").alias("book_count")) \
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

pie_labels = ['Books in Series', 'Standalone Books']
pie_values = [books_with_series, books_without_series]
colors = ['#4A90E2', '#E27D4A']
pie_explode = (0.05, 0)

ax1.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90, 
        colors=colors, explode=pie_explode, shadow=True, textprops={'fontsize': 12})
ax1.set_title('Books: Series vs Standalone', fontsize=14, fontweight='bold')

series_dist_data = series_count_distribution.collect()
series_categories = [row['series_category'] for row in series_dist_data]
series_counts = [row['book_count'] for row in series_dist_data]
series_percentages = [row['percentage'] for row in series_dist_data]

bars = ax2.bar(series_categories, series_counts, color='#4A90E2', edgecolor='black')
ax2.set_xlabel('Series Category', fontsize=12)
ax2.set_ylabel('Book Count', fontsize=12)
ax2.set_title('Distribution by Number of Series', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

for bar, pct in zip(bars, series_percentages):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/series_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Series charts saved to {OUTPUT_DIR}/series_distribution.png")


######################################
# Avg Rating distribution
######################################


total_with_ratings = total_books - missing_stats['missing_average_rating']

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
    .agg(spark_count("*").alias("book_count")) \
    .withColumn("percentage", 
        spark_round((col("book_count") / lit(total_with_ratings)) * 100, 2)
    ) \
    .orderBy("rating_bucket")

print("\n=== AVERAGE RATING DISTRIBUTION ===")
rating_distribution.show()
rating_distribution.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/rating_distribution")

rating_dist_data = rating_distribution.collect()
rating_buckets = [row['rating_bucket'] for row in rating_dist_data]
rating_counts = [row['book_count'] for row in rating_dist_data]

plt.figure(figsize=(10, 6))
plt.bar(rating_buckets, rating_counts, color='#2ECC71', edgecolor='black', width=0.6)
plt.xlabel('Average Rating Range', fontsize=12)
plt.ylabel('Number of Books', fontsize=12)
plt.title('Distribution of Average Ratings (1.0 Intervals)', fontsize=14, fontweight='bold')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rating_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Rating histogram saved to {OUTPUT_DIR}/rating_distribution.png")

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
    .agg(spark_count("*").alias("book_count")) \
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

engagement_dist_data = engagement_distribution.collect()
engagement_levels = [row['engagement_level'] for row in engagement_dist_data]
engagement_counts = [row['book_count'] for row in engagement_dist_data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.bar(engagement_levels, engagement_counts, color='#9B59B6', edgecolor='black')
ax1.set_yscale('log')
ax1.set_xlabel('Engagement Level', fontsize=12)
ax1.set_ylabel('Number of Books (log scale)', fontsize=12)
ax1.set_title('Book Engagement Distribution (Log Scale)', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

mean_val = engagement_stats['mean_ratings']
median_val = engagement_stats['median_ratings']
q1_val = engagement_stats['q1_ratings']
q3_val = engagement_stats['q3_ratings']
min_val = engagement_stats['min_ratings']
max_val = engagement_stats['max_ratings']

bp = ax2.boxplot([q1_val, median_val, q3_val], positions=[1], widths=0.6, 
                  patch_artist=True, vert=True, showfliers=False)
bp['boxes'][0].set_facecolor('#9B59B6')
bp['boxes'][0].set_alpha(0.7)

ax2.plot(1, mean_val, 'r*', markersize=15, label=f'Mean: {mean_val:,.0f}')
ax2.plot(1, median_val, 'go', markersize=10, label=f'Median: {median_val:,.0f}')

ax2.annotate(f'Mean: {mean_val:,.0f}', xy=(1, mean_val), 
            xytext=(1.3, mean_val), fontsize=11, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax2.annotate(f'Median: {median_val:,.0f}', xy=(1, median_val), 
            xytext=(1.3, median_val), fontsize=11, color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

ax2.plot([1, 1], [median_val, mean_val], 'k--', linewidth=2, alpha=0.5)
gap = mean_val - median_val
ax2.text(0.85, (mean_val + median_val) / 2, f'Gap:\n{gap:,.0f}', 
        fontsize=10, ha='right', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.set_ylabel('Ratings Count', fontsize=12)
ax2.set_title('Engagement: Mean vs Median Gap', fontsize=14, fontweight='bold')
ax2.set_xlim(0.5, 2)
ax2.set_xticks([1])
ax2.set_xticklabels(['Engagement\nDistribution'])
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/engagement_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Engagement charts saved to {OUTPUT_DIR}/engagement_distribution.png")

######################################
# Book engagement VS Rating quality
######################################

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
    .agg(spark_count("*").alias("book_count")) \
    .orderBy("rating_quality", "engagement_level")

print("\n=== RATING QUALITY VS ENGAGEMENT ===")
quality_vs_engagement.show(20, truncate=False)
quality_vs_engagement.write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/quality_vs_engagement")

qve_data = quality_vs_engagement.collect()

rating_qualities = ["Low (< 3.0)", "Medium (3.0-4.0)", "High (4.0+)"]
engagement_levels = ["Low (<100)", "Medium (100-1K)", "High (1K-10K)", "Very High (10K+)"]

matrix = np.zeros((len(rating_qualities), len(engagement_levels)))
for row in qve_data:
    r_idx = rating_qualities.index(row['rating_quality']) if row['rating_quality'] in rating_qualities else -1
    e_idx = engagement_levels.index(row['engagement_level']) if row['engagement_level'] in engagement_levels else -1
    if r_idx != -1 and e_idx != -1:
        matrix[r_idx][e_idx] = row['book_count']

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

ax.set_xticks(np.arange(len(engagement_levels)))
ax.set_yticks(np.arange(len(rating_qualities)))
ax.set_xticklabels(engagement_levels)
ax.set_yticklabels(rating_qualities)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Number of Books', rotation=270, labelpad=20)

for i in range(len(rating_qualities)):
    for j in range(len(engagement_levels)):
        text = ax.text(j, i, f'{int(matrix[i, j]):,}',
                      ha="center", va="center", color="black" if matrix[i, j] < matrix.max()/2 else "white",
                      fontsize=11, fontweight='bold')

ax.set_xlabel('Engagement Level', fontsize=12)
ax.set_ylabel('Rating Quality', fontsize=12)
ax.set_title('Rating Quality vs Engagement Level (Book Count)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/quality_vs_engagement_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Heatmap saved to {OUTPUT_DIR}/quality_vs_engagement_heatmap.png")


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
    .agg(spark_count("*").alias("book_count")) \
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
        spark_count("book_id").alias("books_with_shelf"),
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

top_10_by_books = shelf_popularity.limit(10).collect()
shelf_names_books = [row['shelf_name'] for row in top_10_by_books]
books_counts = [row['books_with_shelf'] for row in top_10_by_books]

top_10_by_usage = most_used_shelves.collect()
shelf_names_usage = [row['shelf_name'] for row in top_10_by_usage]
usage_counts = [row['total_shelf_count'] for row in top_10_by_usage]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.barh(shelf_names_books[::-1], books_counts[::-1], color='#3498DB', edgecolor='black')
ax1.set_xlabel('Number of Books', fontsize=12)
ax1.set_ylabel('Shelf Name', fontsize=12)
ax1.set_title('Top 10 Most Popular Shelves (by Book Count)', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3, linestyle='--')

for i, (name, count) in enumerate(zip(shelf_names_books[::-1], books_counts[::-1])):
    ax1.text(count, i, f' {count:,}', va='center', fontsize=10)

ax2.barh(shelf_names_usage[::-1], usage_counts[::-1], color='#E74C3C', edgecolor='black')
ax2.set_xlabel('Total Usage Count', fontsize=12)
ax2.set_ylabel('Shelf Name', fontsize=12)
ax2.set_title('Top 10 Most Used Shelves (by Total Usage)', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3, linestyle='--')

for i, (name, count) in enumerate(zip(shelf_names_usage[::-1], usage_counts[::-1])):
    ax2.text(count, i, f' {count:,}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_shelves.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Top shelves charts saved to {OUTPUT_DIR}/top_shelves.png")

######################################
# Similar Books Analysis
######################################

books_with_similar = df_books.filter((col("similar_books").isNotNull()) & (size(col("similar_books")) > 0)).count()
books_without_similar = total_books - books_with_similar

print("\n=== SIMILAR BOOKS OVERVIEW ===")
print(f"Total Books: {total_books}")
print(f"Books with Similar Books: {books_with_similar} ({books_with_similar/total_books*100:.2f}%)")
print(f"Books without Similar Books: {books_without_similar} ({books_without_similar/total_books*100:.2f}%)")

avg_similar = df_books \
    .filter((col("similar_books").isNotNull()) & (size(col("similar_books")) > 0)) \
    .select(spark_round(mean(size(col("similar_books"))), 2).alias("avg_similar_books")) \
    .collect()[0]

print(f"Average similar books per book: {avg_similar['avg_similar_books']}")

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
    .agg(spark_count("*").alias("book_count")) \
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
        spark_count("*").alias("book_count"),
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

similar_dist_data = similar_books_distribution.collect()
svp_data = similar_vs_performance.collect()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

categories = [row['similar_category'] for row in similar_dist_data]
counts = [row['book_count'] for row in similar_dist_data]
percentages = [row['percentage'] for row in similar_dist_data]

bars = ax1.bar(categories, counts, color='#16A085', edgecolor='black')
ax1.set_xlabel('Similar Books Category', fontsize=12)
ax1.set_ylabel('Number of Books', fontsize=12)
ax1.set_title('Distribution of Similar Books Count', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct}%', ha='center', va='bottom', fontsize=10)

similar_counts = [row['avg_similar_count'] for row in svp_data]
avg_ratings = [row['avg_rating'] for row in svp_data]
avg_engagements = [row['avg_engagement'] for row in svp_data]
book_counts = [row['book_count'] for row in svp_data]
category_labels = [row['similar_category'] for row in svp_data]

scatter = ax2.scatter(similar_counts, avg_ratings, s=[c/50 for c in book_counts], 
                      c=avg_engagements, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1.5)

z = np.polyfit(similar_counts, avg_ratings, 1)
p = np.poly1d(z)
ax2.plot(similar_counts, p(similar_counts), "r--", linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')

for i, label in enumerate(category_labels):
    ax2.annotate(label.replace(' Similar', '').replace('No Similar Books', 'None'), 
                xy=(similar_counts[i], avg_ratings[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax2.set_xlabel('Average Number of Similar Books', fontsize=12)
ax2.set_ylabel('Average Rating', fontsize=12)
ax2.set_title('Similar Books Count vs Rating Performance', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3, linestyle='--')

cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Avg Engagement (Ratings Count)', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/similar_books_analysis.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Similar books analysis charts saved to {OUTPUT_DIR}/similar_books_analysis.png")

most_referenced = df_books \
    .filter((col("similar_books").isNotNull()) & (size(col("similar_books")) > 0)) \
    .select(
        col("book_id").alias("source_book_id"),
        explode(col("similar_books")).alias("similar_book_id")
    ) \
    .groupBy("similar_book_id") \
    .agg(spark_count("source_book_id").alias("referenced_count")) \
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

df_with_desc = df_books.filter(
    (col("description").isNotNull()) & 
    (col("description") != "")
)
total_with_desc = df_with_desc.count()
print(f"Books with descriptions: {total_with_desc}")

keyword_results = []
for keyword in keywords:
    count_with_keyword = df_with_desc.filter(
        lower(col("description")).contains(keyword.lower())
    ).count()
    
    percentage = (count_with_keyword / total_with_desc * 100) if total_with_desc > 0 else 0
    keyword_results.append((keyword, count_with_keyword, round(percentage, 2)))

keyword_results.sort(key=lambda x: x[1], reverse=True)

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


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

keyword_freq = {kw: kw_count for kw, kw_count, pct in keyword_results}
wordcloud = WordCloud(width=800, height=400, background_color='white', 
                      colormap='viridis', relative_scaling=0.5,
                      min_font_size=10).generate_from_frequencies(keyword_freq)

ax1.imshow(wordcloud, interpolation='bilinear')
ax1.axis('off')
ax1.set_title('Keyword Frequency Word Cloud', fontsize=14, fontweight='bold', pad=20)

top_keywords = keyword_results[:15]
kw_names = [kw for kw, kw_count, pct in top_keywords]
kw_counts = [kw_count for kw, kw_count, pct in top_keywords]

ax2.barh(kw_names[::-1], kw_counts[::-1], color='#E67E22', edgecolor='black')
ax2.set_xlabel('Number of Books', fontsize=12)
ax2.set_ylabel('Keyword', fontsize=12)
ax2.set_title('Top 15 Keywords in Book Descriptions', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3, linestyle='--')

for i, (kw, kw_count) in enumerate(zip(kw_names[::-1], kw_counts[::-1])):
    ax2.text(kw_count, i, f' {kw_count:,}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/keyword_analysis.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Keyword analysis charts saved to {OUTPUT_DIR}/keyword_analysis.png")

multi_keyword_presence = df_with_desc.select("book_id", "description")
for keyword in keywords:
    multi_keyword_presence = multi_keyword_presence.withColumn(
        f"has_{keyword}",
        when(lower(col("description")).contains(keyword.lower()), 1).otherwise(0)
    )

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
    .agg(spark_count("*").alias("book_count")) \
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

kw_dist_data = keyword_count_distribution.collect()
kw_categories = [row['keyword_category'] for row in kw_dist_data]
kw_book_counts = [row['book_count'] for row in kw_dist_data]
kw_percentages = [row['percentage'] for row in kw_dist_data]

plt.figure(figsize=(12, 6))
x_positions = range(len(kw_categories))

plt.fill_between(x_positions, 0, kw_book_counts, alpha=0.7, color='#3498DB', edgecolor='black', linewidth=2)

plt.xlabel('Keyword Density Category', fontsize=12)
plt.ylabel('Number of Books', fontsize=12)
plt.title('Distribution of Keyword Counts per Book Description', fontsize=14, fontweight='bold')
plt.xticks(x_positions, kw_categories, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')

for i, (cat, count, pct) in enumerate(zip(kw_categories, kw_book_counts, kw_percentages)):
    plt.text(i, count, f'{count:,}\n({pct}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/keyword_density_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Keyword density chart saved to {OUTPUT_DIR}/keyword_density_distribution.png")

avg_keywords = multi_keyword_presence.select(
    spark_round(mean("total_keywords"), 2).alias("avg_keywords_per_book")
).collect()[0]
print(f"\nAverage keywords per book description: {avg_keywords['avg_keywords_per_book']}")

spark.stop()