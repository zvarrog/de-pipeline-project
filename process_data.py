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
# если смонтирован/добавлен
ORIGINAL_FILE_PATH = "/app/data/original/kindle_reviews.csv"
DOWNLOAD_DEST = "/tmp/kindle_reviews.csv"  # временное место для загрузки по URL
DOWNLOAD_ZIP_DEST = "/tmp/kindle_reviews.zip"  # для zip файлов
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
    1) Наличие ORIGINAL_FILE_PATH - использовать локальный полный файл
    2) Автоматическая загрузка с Kaggle -> DOWNLOAD_DEST
    3) Если ничего не получается - ошибка
    """
    # Проверяем наличие локального полного файла
    if os.path.exists(ORIGINAL_FILE_PATH):
        print("[process_data] Using ORIGINAL_FILE_PATH (local full dataset)")
        return ORIGINAL_FILE_PATH

    # Пытаемся загрузить с Kaggle
    try:
        print("[process_data] Local full dataset not found, downloading from Kaggle...")
        return download_from_kaggle()
    except Exception as e:
        print(f"[process_data] Kaggle download failed: {e}")
        raise RuntimeError(
            "Neither local dataset nor Kaggle download available. "
            "Please either:\n"
            "1) Place kindle_reviews.csv in data/original/, or\n"
            "2) Configure Kaggle API credentials (~/.kaggle/kaggle.json)"
        ) from e


def download_from_kaggle() -> str:
    """Автоматически загружает датасет с Kaggle."""
    import zipfile

    try:
        import kaggle
    except ImportError:
        raise RuntimeError(
            "Kaggle module not available. Install with: pip install kaggle")

    # Проверяем кеш
    cache_file = Path(DOWNLOAD_DEST)
    if cache_file.exists():
        print(f"[process_data] Using cached Kaggle file: {cache_file}")
        return str(cache_file)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    zip_path = Path(DOWNLOAD_ZIP_DEST)

    print("[process_data] Downloading dataset from Kaggle (bharadwaj6/kindle-reviews)...")
    print("[process_data] This may take a few minutes for the first download...")

    try:
        # Загружаем ZIP с Kaggle
        kaggle.api.dataset_download_files(
            'bharadwaj6/kindle-reviews',
            path=zip_path.parent,
            unzip=False
        )

        # Переименовываем скачанный файл
        downloaded_zip = zip_path.parent / "kindle-reviews.zip"
        if downloaded_zip.exists():
            downloaded_zip.rename(zip_path)

        # Извлекаем CSV
        print("[process_data] Extracting CSV from ZIP archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            csv_files = [f for f in zip_ref.namelist(
            ) if f.lower().endswith('.csv')]
            if not csv_files:
                raise RuntimeError("No CSV file found in Kaggle ZIP archive")

            csv_filename = csv_files[0]  # обычно kindle_reviews.csv
            print(f"[process_data] Extracting {csv_filename}")

            with zip_ref.open(csv_filename) as source, open(cache_file, "wb") as target:
                target.write(source.read())

        # Удаляем ZIP после извлечения
        zip_path.unlink()

        size_mb = cache_file.stat().st_size / (1024 * 1024)
        print(
            f"[process_data] Kaggle dataset ready: {cache_file} ({size_mb:.1f} MB)")

        return str(cache_file)

    except Exception as e:
        # Очищаем частично загруженные файлы
        for file_path in [zip_path, cache_file]:
            if file_path.exists():
                file_path.unlink()
        raise RuntimeError(f"Failed to download from Kaggle: {e}")


def download_dataset(url: str) -> str:
    """Скачивает датасет по URL. Поддерживает обычные файлы и ZIP архивы.
    Эта функция сохранена для совместимости с внешними URL."""
    import zipfile

    # Определяем тип файла по URL
    is_zip = url.lower().endswith('.zip')
    dest = Path(DOWNLOAD_ZIP_DEST if is_zip else DOWNLOAD_DEST)

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Скачиваем если файла ещё нет
    if not dest.exists():
        print(f"[process_data] Downloading dataset from {url} -> {dest}")
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
    else:
        print(f"[process_data] Using cached downloaded file: {dest}")

    # Если это ZIP файл, извлекаем CSV
    if is_zip:
        csv_path = Path("/tmp/kindle_reviews.csv")
        if not csv_path.exists():
            print(f"[process_data] Extracting CSV from ZIP: {dest}")
            try:
                with zipfile.ZipFile(dest, 'r') as zip_ref:
                    csv_files = [f for f in zip_ref.namelist(
                    ) if f.lower().endswith('.csv')]
                    if not csv_files:
                        raise RuntimeError("No CSV file found in ZIP archive")

                    csv_filename = csv_files[0]
                    print(f"[process_data] Extracting {csv_filename} from ZIP")

                    with zip_ref.open(csv_filename) as source, open(csv_path, "wb") as target:
                        target.write(source.read())

            except Exception as e:
                raise RuntimeError(f"Failed to extract CSV from ZIP: {e}")

        return str(csv_path)

    return str(dest)


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

df = df.withColumn("review_timestamp", to_date(
    from_unixtime(col("unixReviewTime"))))
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
        "reviewerID_helpful_ratio_avg", avg(
            "helpful_ratio").over(reviewer_window)
    )
)

df = df.drop(
    "helpful", "_c0", "reviewerName", "unixReviewTime", "helpful_votes", "total_votes"
)

df.write.mode("overwrite").parquet("/app/output/reviews_processed.parquet")

spark.stop()
print("Обработка данных завершена")
