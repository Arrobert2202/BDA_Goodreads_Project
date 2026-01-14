from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, collect_list, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, LDA
from pyspark.ml import Pipeline

SAMPLE_BASE = "hdfs:///user/ubuntu/proiect_bda/output_eda/sample_json"
FULL_BASE = "hdfs:///user/ubuntu/proiect_bda/output_eda/parquet"

OUTPUT_DIR = FULL_BASE + "/output_parquet_clusters"


keywords = [
    "love","war","mystery","fantasy","magic","science","history",
    "biography","business","psychology","self-help",
    "adventure","journey","family","friendship","survival",
    "dark","emotional","inspiring","funny",
    "hero","villain","detective",
    "young","adult","children",
    "future","past","world","society",
    "bestselling","award"]


spark = SparkSession.builder \
    .appName("Goodreads_Clustering") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

df_books = spark.read.parquet(f"{FULL_BASE}")


# =============================================================================
# APPROACH 1: SHELF-BASED TOPIC EXTRACTION
# Uses popular_shelves as topics/genres
# =============================================================================
def shelf_based_topics(df):
    """
    Extract topics from popular_shelves - these are user-generated tags
    that effectively represent book genres/topics.
    """
    # Explode shelves to get one row per shelf
    df_shelves = df.select(
        col("book_id"),
        col("title"),
        explode(col("popular_shelves")).alias("shelf")
    )
    
    # Extract shelf name and count
    df_shelves = df_shelves.select(
        "book_id",
        "title",
        col("shelf.name").alias("shelf_name"),
        col("shelf.count").cast("int").alias("shelf_count")
    )
    
    # Filter out generic shelves, keep genre-related ones
    generic_shelves = ["to-read", "currently-reading", "owned", "favorites", 
                       "books-i-own", "wish-list", "to-buy", "default"]
    
    df_topics = df_shelves.filter(~col("shelf_name").isin(generic_shelves))
    
    # Get top shelf (topic) per book
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number
    
    window = Window.partitionBy("book_id").orderBy(col("shelf_count").desc())
    df_top_topic = df_topics.withColumn("rank", row_number().over(window)) \
                            .filter(col("rank") <= 3) \
                            .groupBy("book_id", "title") \
                            .agg(collect_list("shelf_name").alias("topics"))
    
    return df_top_topic


# =============================================================================
# APPROACH 2: KEYWORDS CLUSTERING
# Cluster by keyword presence in description
# =============================================================================

def keyword_based_clustering(df, keywords):
    """
    Cluster books based on presence of predefined keywords in their descriptions.
    Useful for thematic clustering (e.g., books about love, war, mystery, etc.).
    """
    # Prepare text data
    df_text = df.select(
        col("book_id"),
        col("title"),
        lower(regexp_replace(col("description"), "[^a-zA-Z\\s]", "")).alias("description_clean")
    ).filter(col("description_clean").isNotNull() & (col("description_clean") != ""))
    
    # Create keyword presence columns
    for keyword in keywords:
        df_text = df_text.withColumn(
            f"kw_{keyword}",
            (col("description_clean").contains(keyword)).cast("int")
        )
    
    feature_cols = [f"kw_{kw}" for kw in keywords]
    
    # Assemble features into vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    kmeans = KMeans(k=10, seed=42, featuresCol="features", predictionCol="cluster")
    
    pipeline = Pipeline(stages=[assembler, kmeans])
    model = pipeline.fit(df_text)
    df_clustered = model.transform(df_text)
    
    return df_clustered.select("book_id", "title", "cluster"), model


# =============================================================================
# APPROACH 3: FEATURE-BASED CLUSTERING
# Cluster by numerical attributes (ratings, pages, popularity)
# =============================================================================
def feature_based_clustering(df, num_clusters=10):
    """
    Cluster books based on numerical features like ratings, page count, etc.
    Good for finding book "profiles" (e.g., popular long books vs niche short books).
    """
    df_features = df.select(
        col("book_id"),
        col("title"),
        col("average_rating").cast("double").alias("avg_rating"),
        col("num_pages").cast("double").alias("num_pages"),
        col("ratings_count").cast("double").alias("ratings_count"),
        col("text_reviews_count").cast("double").alias("reviews_count")
    ).na.fill(0)
    
    # Assemble features into vector
    assembler = VectorAssembler(
        inputCols=["avg_rating", "num_pages", "ratings_count", "reviews_count"],
        outputCol="features_raw"
    )
    
    # Scale features (important for K-Means)
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    
    kmeans = KMeans(k=num_clusters, seed=42, featuresCol="features", predictionCol="cluster")
    
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    model = pipeline.fit(df_features)
    df_clustered = model.transform(df_features)
    
    return df_clustered.select("book_id", "title", "avg_rating", "num_pages", "ratings_count", "cluster"), model


# =============================================================================
# APPROACH 4: DESCRIPTION-BASED CLUSTERING (TF-IDF + K-Means)
# NLP-based topic extraction from book descriptions
# =============================================================================
def description_clustering(df, num_clusters=10):
    """
    Cluster books based on their description text using TF-IDF and K-Means.
    """
    # Prepare text data
    df_text = df.select(
        col("book_id"),
        col("title"),
        lower(regexp_replace(col("description"), "[^a-zA-Z\\s]", "")).alias("description_clean")
    ).filter(col("description_clean").isNotNull() & (col("description_clean") != ""))
    
    # NLP Pipeline: Tokenize -> Remove stopwords -> TF-IDF
    tokenizer = Tokenizer(inputCol="description_clean", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    kmeans = KMeans(k=num_clusters, seed=42, featuresCol="features", predictionCol="cluster")
    
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, kmeans])
    
    model = pipeline.fit(df_text)
    df_clustered = model.transform(df_text)
    
    return df_clustered.select("book_id", "title", "cluster"), model


# =============================================================================
# APPROACH 5: LDA TOPIC MODELING
# Latent Dirichlet Allocation for discovering latent topics
# =============================================================================
def lda_topic_modeling(df, num_topics=10):
    """
    Apply LDA to discover latent topics from book descriptions.
    """
    df_text = df.select(
        col("book_id"),
        col("title"),
        lower(regexp_replace(col("description"), "[^a-zA-Z\\s]", "")).alias("description_clean")
    ).filter(col("description_clean").isNotNull() & (col("description_clean") != ""))
    
    # Tokenize and vectorize
    tokenizer = Tokenizer(inputCol="description_clean", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    cv = HashingTF(inputCol="filtered_words", outputCol="features", numFeatures=1000)
    
    # Apply transformations
    df_tokens = tokenizer.transform(df_text)
    df_filtered = stopwords_remover.transform(df_tokens)
    df_vectors = cv.transform(df_filtered)
    
    # Fit LDA model
    lda = LDA(k=num_topics, maxIter=20, seed=42)
    lda_model = lda.fit(df_vectors)
    
    # Get topic distribution for each document
    df_topics = lda_model.transform(df_vectors)
    
    # Get top words per topic (for interpretation)
    topics = lda_model.describeTopics(maxTermsPerTopic=10)
    
    return df_topics.select("book_id", "title", "topicDistribution"), topics, lda_model





# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TOPIC EXTRACTION & CLUSTERING FOR GOODREADS BOOKS")
    print("=" * 60)
    
    print("\n[1] SHELF-BASED TOPIC EXTRACTION")
    df_shelf_topics = shelf_based_topics(df_books)
    df_shelf_topics.show(10, truncate=False)

    df_shelf_topics.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/shelf_topics")


    print("\n[2] KEYWORD-BASED CLUSTERING")
    df_keyword_clusters, keyword_model = keyword_based_clustering(df_books, keywords)
    df_keyword_clusters.show(10)
    df_keyword_clusters.groupBy("cluster").count().orderBy("cluster").show()

    df_keyword_clusters.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/keyword_clusters")


    print("\n[3] FEATURE-BASED CLUSTERING")
    df_feature_clusters, feature_model = feature_based_clustering(df_books, num_clusters=8)
    df_feature_clusters.show(10)
    df_feature_clusters.groupBy("cluster").count().orderBy("cluster").show()

    df_feature_clusters.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/feature_clusters")

    
    print("\n[4] DESCRIPTION-BASED CLUSTERING (TF-IDF + K-Means)")
    df_desc_clusters, kmeans_model = description_clustering(df_books, num_clusters=8)
    df_desc_clusters.show(10)
    df_desc_clusters.groupBy("cluster").count().orderBy("cluster").show()

    df_desc_clusters.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/description_clusters")    

    
    print("\n[5] LDA TOPIC MODELING")
    df_lda, topic_words, lda_model = lda_topic_modeling(df_books, num_topics=8)
    print("Topic word indices (use vocabulary to map to actual words):")
    topic_words.show(truncate=False)
    
    df_lda.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/lda_topics")
    
    
    print("\n✓ Results saved to:", OUTPUT_DIR)
    spark.stop()