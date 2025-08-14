import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import os
import pandas as pd


class SparkDataProcessor:
    
    def __init__(self, app_name="KindleReviewsProcessing"):
        self.spark = (SparkSession.builder
                     .appName(app_name)
                     .config("spark.sql.adaptive.enabled", "true")
                     .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                     .config("spark.driver.memory", "4g")
                     .config("spark.driver.maxResultSize", "2g")
                     .getOrCreate())
        
        self.spark.sparkContext.setLogLevel("WARN")
        print(f"Spark session created: {self.spark.version}")
    
    def load_data(self, file_path):
        print(f"Loading data from {file_path}")
        
        schema = StructType([
            StructField("asin", StringType(), True),
            StructField("helpful", StringType(), True),
            StructField("overall", DoubleType(), True),
            StructField("reviewText", StringType(), True),
            StructField("reviewTime", StringType(), True),
            StructField("reviewerID", StringType(), True),
            StructField("reviewerName", StringType(), True),
            StructField("summary", StringType(), True),
            StructField("unixReviewTime", LongType(), True)
        ])
        
        df = (self.spark.read
              .option("header", "true")
              .option("multiline", "true")
              .option("escape", '"')
              .schema(schema)
              .csv(file_path))
        
        print(f"Loaded {df.count():,} records")
        return df
    
    def clean_data(self, df):
        print("Cleaning data...")
        
        df_clean = df.filter(
            (col("reviewText").isNotNull()) & 
            (col("overall").isNotNull()) &
            (length(col("reviewText")) > 10)
        )
        
        df_clean = df_clean.withColumn("rating", col("overall").cast("integer"))
        
        df_clean = df_clean.withColumn(
            "clean_text",
            regexp_replace(
                regexp_replace(col("reviewText"), "<[^>]*>", ""),
                "[^a-zA-Z0-9\\s]", " "
            )
        )
        
        df_clean = df_clean.withColumn("clean_text", lower(col("clean_text")))
        
        print(f"After cleaning: {df_clean.count():,} records")
        return df_clean
    
    def feature_engineering(self, df):
        print("Creating features...")
        
        df = df.withColumn("text_length", length(col("clean_text")))
        df = df.withColumn("word_count", size(split(col("clean_text"), "\\s+")))
        df = df.withColumn("exclamation_count", 
                          size(split(col("reviewText"), "!")) - 1)
        df = df.withColumn("question_count",
                          size(split(col("reviewText"), "\\?")) - 1)
        df = df.withColumn("review_month", 
                          month(to_date(col("reviewTime"), "MM dd, yyyy")))
        
        return df
    
    def analyze_data(self, df):
        print("Analyzing data...")
        
        print("Rating distribution:")
        rating_dist = df.groupBy("rating").count().orderBy("rating")
        rating_dist.show()
        
        print("Text length statistics:")
        df.select(
            avg("text_length").alias("avg_length"),
            min("text_length").alias("min_length"),
            max("text_length").alias("max_length"),
            stddev("text_length").alias("std_length")
        ).show()
        
        return df
    
    def prepare_ml_features(self, df, sample_size=100000):
        print("Preparing ML features...")
        
        if df.count() > sample_size:
            df_sample = df.sample(fraction=sample_size/df.count(), seed=42)
        else:
            df_sample = df
        
        print(f"Sample size for ML: {df_sample.count():,}")
        
        tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
        stop_words_remover = StopWordsRemover(
            inputCol="words", 
            outputCol="filtered_words"
        )
        hashing_tf = HashingTF(
            numFeatures=10000, 
            inputCol="filtered_words", 
            outputCol="tf_features"
        )
        idf = IDF(
            inputCol="tf_features", 
            outputCol="tfidf_features"
        )
        
        pipeline = Pipeline(stages=[
            tokenizer,
            stop_words_remover, 
            hashing_tf,
            idf
        ])
        
        pipeline_model = pipeline.fit(df_sample)
        df_features = pipeline_model.transform(df_sample)
        
        df_ml = df_features.select(
            "rating",
            "tfidf_features",
            "text_length",
            "word_count",
            "exclamation_count",
            "question_count"
        ).filter(col("rating").isNotNull())
        
        return df_ml, pipeline_model
    
    def train_spark_model(self, df_ml):
        print("Training Spark model...")
        
        train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)
        
        print(f"Training samples: {train_df.count():,}")
        print(f"Test samples: {test_df.count():,}")
        
        lr = SparkLogisticRegression(
            featuresCol="tfidf_features",
            labelCol="rating",
            maxIter=100,
            regParam=0.01
        )
        
        model = lr.fit(train_df)
        predictions = model.transform(test_df)
        
        evaluator = MulticlassClassificationEvaluator(
            labelCol="rating",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        accuracy = evaluator.evaluate(predictions)
        print(f"Spark model accuracy: {accuracy:.3f}")
        
        return model, accuracy
    
    def save_processed_data(self, df, output_path):
        print(f"Saving to {output_path}")
        
        df.coalesce(1).write.mode("overwrite").parquet(output_path)
        
        df_sample = df.limit(1000)
        df_sample.toPandas().to_csv(f"{output_path}/sample.csv", index=False)
    
    def stop(self):
        self.spark.stop()


def main():
    processor = SparkDataProcessor()
    
    try:
        df = processor.load_data("data/original/kindle_reviews.csv")
        df_clean = processor.clean_data(df)
        df_features = processor.feature_engineering(df_clean)
        df_analyzed = processor.analyze_data(df_features)
        df_ml, pipeline_model = processor.prepare_ml_features(df_analyzed)
        spark_model, accuracy = processor.train_spark_model(df_ml)
        processor.save_processed_data(df_features, "data/processed")
        
        print(f"Processing completed. Accuracy: {accuracy:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        processor.stop()


if __name__ == "__main__":
    main()
