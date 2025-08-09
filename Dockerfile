FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

RUN curl -o spark.tgz "https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz" && \
    tar -xzf spark.tgz -C /opt/ && \
    mv /opt/spark-3.5.0-bin-hadoop3 /opt/spark && \
    rm spark.tgz

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY process_data.py .
COPY kindle_reviews.csv .

CMD ["python", "process_data.py"]