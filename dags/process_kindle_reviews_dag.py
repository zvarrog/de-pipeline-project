from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

"""
Важно для Windows/Docker Desktop:
DockerOperator создаёт НОВЫЙ контейнер через Docker daemon на хосте.
Bind source path ДОЛЖЕН указывать путь, существующий на ХОСТЕ, а не внутри контейнера Airflow.
Поэтому используем абсолютный Windows-путь проекта: C:/Users/dasiqe/de-pipeline-project
и передаём его через volumes в формате host_path:container_path.
Внутренние пути /opt/airflow/* здесь не подойдут как source.
"""

HOST_PROJECT_PATH = "C:/Users/dasiqe/de-pipeline-project"  # абсолютный путь проекта на хосте
OUTPUT_PATH = f"{HOST_PROJECT_PATH}/output"  # монтируем только выход – входной sample уже внутри образа


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

    # Единственная задача: Spark обработка (sample CSV встроен в образ; full data можно будет добавить позже)
    spark_processing_task = DockerOperator(
        task_id="spark_data_processing",
        image="kindle-reviews-processor:latest",
        auto_remove="success",  # удаляем контейнер после успешного завершения
        mounts=[
            Mount(source=OUTPUT_PATH, target="/app/output", type="bind"),
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,  # избегаем лишнего tmp bind на Windows
        environment={
            "PYTHONUNBUFFERED": "1",
        },
    )
