from __future__ import annotations
import os
import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from docker.types import Mount

# Импортируем специальный класс Mount для описания томов
from docker.types import (
    Mount,
)  # (оставляем импорт если захотим вернуться к Mount, но ниже используем volumes)

"""
DockerOperator создаёт НОВЫЙ контейнер через Docker daemon на хосте.
Bind source path ДОЛЖЕН указывать путь, существующий на ХОСТЕ, а не внутри контейнера Airflow.
Для Linux/Codespace используем путь проекта в workspace.
"""

HOST_PROJECT_PATH = "/workspaces/de-pipeline-project"
OUTPUT_PATH = f"{HOST_PROJECT_PATH}/output"


def check_inputs():
    # Проверка не нужна для DockerOperator, так как он создаёт отдельный контейнер
    # Все необходимые проверки выполняются внутри контейнера обработки данных
    print("Проверка входных данных пропущена - DockerOperator создаёт отдельный контейнер")


# Создаем объекты Mount для каждого пробрасываемого тома
# Это более явный и современный способ
# Оставляем определение через volumes (проще на Windows). Если нужно вернуться к Mount – можно раскомментировать.
# data_mount = Mount(source=f"{HOST_PROJECT_PATH}/data", target="/app/data", type="bind", read_only=True)
# output_mount = Mount(source=f"{HOST_PROJECT_PATH}/output", target="/app/output", type="bind")


with DAG(
    dag_id="end_to_end_kindle_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["portfolio", "spark", "pytorch"],
) as dag:

    check_input_task = PythonOperator(
        task_id="check_input_paths",
        python_callable=check_inputs,
    )

    # ЗАДАЧА: Обработка данных с помощью PySpark в Docker
    spark_processing_task = DockerOperator(
        task_id="spark_data_processing",
        image="kindle-reviews-processor:latest",
        auto_remove="success",
        mounts=[
            Mount(source=OUTPUT_PATH, target="/app/output", type="bind"),
            Mount(source=f"{HOST_PROJECT_PATH}/data/original",
                  target="/app/data/original", type="bind", read_only=True),
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,
        environment={
            "PYTHONUNBUFFERED": "1",
        },
    )
    check_input_task >> spark_processing_task
