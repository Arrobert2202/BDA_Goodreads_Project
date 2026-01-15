from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("Goodreads-ALS-Recommender").getOrCreate()

INPUT = "hdfs:///user/ubuntu/proiect_bda/output_recommender/parquet"
OUT   = "hdfs:///user/ubuntu/proiect_bda/output_recommender_results"

df = spark.read.parquet(INPUT).select("user_id", "book_id", "rating", "title")

df = (df.filter(F.col("rating").isNotNull())
        .withColumn("rating", F.col("rating").cast("double"))
        .filter(F.col("rating") > 0))

df = (df.groupBy("user_id", "book_id")
        .agg(F.avg("rating").alias("rating"),
             F.first("title", ignorenulls=True).alias("title")))

u_cnt = df.groupBy("user_id").count().withColumnRenamed("count","u_cnt")
b_cnt = df.groupBy("book_id").count().withColumnRenamed("count","b_cnt")

df = (df.join(u_cnt, "user_id")
        .join(b_cnt, "book_id")
        .filter((F.col("u_cnt") >= 5) & (F.col("b_cnt") >= 10))
        .drop("u_cnt","b_cnt"))

user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="skip")
book_indexer = StringIndexer(inputCol="book_id", outputCol="book_idx", handleInvalid="skip")

df_i = user_indexer.fit(df).transform(df)
df_i = book_indexer.fit(df_i).transform(df_i)

df_i = (df_i.withColumn("user_idx", F.col("user_idx").cast("int"))
            .withColumn("book_idx", F.col("book_idx").cast("int")))

train, test = df_i.randomSplit([0.8, 0.2], seed=42)

als = ALS(
    userCol="user_idx",
    itemCol="book_idx",
    ratingCol="rating",
    implicitPrefs=False,
    nonnegative=True,
    coldStartStrategy="drop",
    rank=50,
    regParam=0.1,
    maxIter=10
)

model = als.fit(train)

pred = model.transform(test)
rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction").evaluate(pred)

spark.createDataFrame([(float(rmse), 50, 0.1, 10)], ["rmse","rank","regParam","maxIter"]) \
     .write.mode("overwrite").json(OUT + "/als_metrics")

recs = model.recommendForAllUsers(10)

recs_expl = (recs
    .withColumn("rec", F.explode("recommendations"))
    .select("user_idx",
            F.col("rec.book_idx").alias("book_idx"),
            F.col("rec.rating").alias("score")))

book_lookup = df_i.select("book_idx","book_id","title").dropDuplicates(["book_idx"])

(recs_expl.join(book_lookup, "book_idx", "left")
         .orderBy("user_idx", F.desc("score"))
         .write.mode("overwrite")
         .parquet(OUT + "/als_recommendations"))

print("ALS RMSE =", rmse)

spark.stop()
