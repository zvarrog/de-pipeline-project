from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    udf,
    regexp_extract,
    to_timestamp,
    datediff,
    lit,
    length,
    lower,
    regexp_replace,
    when,
    isnan,
    isnull,
    current_timestamp,
    from_unixtime,
    to_date,
    avg,
    count,
)
from pyspark.sql.types import StringType, IntegerType, FloatType
from pyspark.sql.window import Window


FILE_PATH = "/app/kindle_reviews.csv"
TEXT_COLS = ["reviewText", "summary"]
AGG_COLS = ["asin", "reviewerID"]

# Дополнительные настройки для оптимизации работы
spark = (
    SparkSession.builder.appName("KindleReviewsTransformation")
    .master("local[2]")
    .config("spark.driver.memory", "3g")
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.executor.memory", "2g")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .getOrCreate()
)

# Опции парсинга, необходимые из-за особых символов в summary и reviewText
df = (
    spark.read.option("multiline", "true")
    .option("escape", '"')
    .csv(FILE_PATH, header=True, inferSchema=True)
)


df = df.withColumn(
    "helpful_votes",
    regexp_extract(col("helpful"), r"\[(\d+), (\d+)\]", 1).cast(IntegerType()),
).withColumn(
    "total_votes",
    regexp_extract(col("helpful"), r"\[(\d+), (\d+)\]", 2).cast(IntegerType()),
)

df = df.withColumn(
    "helpful_ratio",
    when((col("total_votes") == 0) | isnull(col("total_votes")), 0).otherwise(
        col("helpful_votes") / col("total_votes")
    ),
)

df = df.withColumn("review_timestamp", to_date(from_unixtime(col("unixReviewTime"))))
df = df.withColumn(
    "days_since_review", datediff(current_timestamp(), col("review_timestamp"))
)

for column in TEXT_COLS:
    df = df.withColumn(
        column + "_clean",
        when(col(column).isNull(), "").otherwise(
            regexp_replace(lower(col(column)), r"[^a-z0-9\s]", "")
        ),
    )
    df = df.withColumn(column + "_len", length(col(column)))

asin_window = Window.partitionBy("asin")
reviewer_window = Window.partitionBy("reviewerID")

df = (
    df.withColumn("asin_overall_avg", avg("overall").over(asin_window))
    .withColumn("reviewerID_overall_avg", avg("overall").over(reviewer_window))
    .withColumn("asin_total_votes_count", count("total_votes").over(asin_window))
    .withColumn(
        "reviewerID_helpful_ratio_avg", avg("helpful_ratio").over(reviewer_window)
    )
)

df = df.drop(
    "helpful", "_c0", "reviewerName", "unixReviewTime", "helpful_votes", "total_votes"
)

df.write.mode("overwrite").parquet("/app/output/reviews_processed.parquet")

spark.stop()
print("Обработка данных завершена")
