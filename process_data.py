import os
from pathlib import Path
import requests
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


# Пути по умолчанию внутри контейнера
ORIGINAL_FILE_PATH = "/app/data/original/kindle_reviews.csv"  # если смонтирован/добавлен
SAMPLE_FILE_PATH = "/app/data/sample/kindle_reviews_sample.csv"  # baked-in семпл
DOWNLOAD_DEST = "/tmp/kindle_reviews.csv"  # временное место для загрузки по URL
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

def resolve_input_path() -> str:
    """Определяет входной путь к датасету.
    Приоритет:
    1) Переменная окружения DATA_URL -> скачиваем в DOWNLOAD_DEST
    2) DATA_MODE=full и наличие ORIGINAL_FILE_PATH
    3) SAMPLE_FILE_PATH (встроенный семпл в образ)
    """
    data_url = os.getenv("DATA_URL", "").strip()
    data_mode = os.getenv("DATA_MODE", "sample").strip().lower()

    if data_url:
        # Скачиваем файл только если ещё нет
        dest = Path(DOWNLOAD_DEST)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"[process_data] Downloading dataset from {data_url} -> {dest}")
            try:
                with requests.get(data_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset: {e}")
        else:
            print(f"[process_data] Using cached downloaded file: {dest}")
        return str(dest)

    if data_mode == "full" and os.path.exists(ORIGINAL_FILE_PATH):
        print("[process_data] Using ORIGINAL_FILE_PATH")
        return ORIGINAL_FILE_PATH

    print("[process_data] Using SAMPLE_FILE_PATH (built-in)")
    return SAMPLE_FILE_PATH


INPUT_PATH = resolve_input_path()

# Опции парсинга, необходимые из-за особых символов в summary и reviewText
df = (
    spark.read.option("multiline", "true")
    .option("escape", '"')
    .csv(INPUT_PATH, header=True, inferSchema=True)
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
