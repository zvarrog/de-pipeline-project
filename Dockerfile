FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y openjdk-21-jre-headless curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

RUN curl -o spark.tgz "https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz" && \
    tar -xzf spark.tgz -C /opt/ && \
    mv /opt/spark-3.5.0-bin-hadoop3 /opt/spark && \
    rm spark.tgz

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY process_data.py .
COPY ml_pipeline.py .
COPY ml_pipeline_enhanced.py .
COPY ml_pipeline_lstm.py .
COPY ml_pipeline_corrected.py .
COPY ml_pipeline_advanced.py .
COPY ml_pipeline_embeddings.py .
COPY ml_pipeline_meta.py .
COPY ml_report.py .
COPY final_comparison.py .
COPY data_quality.py .
COPY api_server.py .
COPY model_registry.py .
COPY test_api.py .

# Create necessary directories
RUN mkdir -p /app/data/processed /app/logs /app/models /app/mlruns

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "api_server.py"]